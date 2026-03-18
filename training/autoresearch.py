#!/usr/bin/env python3
"""
Autoresearch: Autonomous experiment loop for hybrid corner detector.

Systematically explores hyperparams/architecture by patching train_corners_hybrid.py,
running short training runs, and keeping improvements.

Usage:
    python3 training/autoresearch.py [--epochs 10] [--timeout 1200]
"""

import atexit
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_SCRIPT = BASE_DIR / "train_corners_hybrid.py"
BACKUP_FILE = BASE_DIR / ".autoresearch_backup"
TSV_FILE = BASE_DIR / "autoresearch_results.tsv"
MODELS_DIR = BASE_DIR / "models"

# Each experiment: {id, name, patches: [(old, new), ...], batch_size_override}
EXPERIMENTS = [
    # --- Learning rate ---
    {
        "id": "001", "name": "lr=0.0003",
        "patches": [
            ('parser.add_argument("--lr", type=float, default=0.001)',
             'parser.add_argument("--lr", type=float, default=0.0003)'),
        ],
    },
    {
        "id": "002", "name": "lr=0.003",
        "patches": [
            ('parser.add_argument("--lr", type=float, default=0.001)',
             'parser.add_argument("--lr", type=float, default=0.003)'),
        ],
    },
    {
        "id": "003", "name": "lr=0.0001",
        "patches": [
            ('parser.add_argument("--lr", type=float, default=0.001)',
             'parser.add_argument("--lr", type=float, default=0.0001)'),
        ],
    },
    # --- Dropout ---
    {
        "id": "004", "name": "dropout=0.4",
        "patches": [
            ('nn.Dropout(0.2),', 'nn.Dropout(0.4),'),
        ],
    },
    {
        "id": "005", "name": "dropout=0.0",
        "patches": [
            ('nn.Dropout(0.2),', 'nn.Dropout(0.0),'),
        ],
    },
    # --- Regression head ---
    {
        "id": "006", "name": "512->256->8 head",
        "patches": [
            ("""    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 8),
        nn.Sigmoid(),
    )""",
             """    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 256),
        nn.ReLU(),
        nn.Linear(256, 8),
        nn.Sigmoid(),
    )"""),
        ],
    },
    {
        "id": "007", "name": "512->256->64->8 head",
        "patches": [
            ("""    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 8),
        nn.Sigmoid(),
    )""",
             """    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
        nn.Sigmoid(),
    )"""),
        ],
    },
    # --- Loss functions ---
    {
        "id": "008", "name": "MSELoss",
        "patches": [
            ('criterion = nn.SmoothL1Loss()',
             'criterion = nn.MSELoss()'),
        ],
    },
    {
        "id": "009", "name": "SmoothL1(beta=0.5)",
        "patches": [
            ('criterion = nn.SmoothL1Loss()',
             'criterion = nn.SmoothL1Loss(beta=0.5)'),
        ],
    },
    # --- Weight decay ---
    {
        "id": "010", "name": "weight_decay=0.05",
        "patches": [
            ('weight_decay=0.001)',
             'weight_decay=0.05)'),
        ],
    },
    {
        "id": "011", "name": "weight_decay=0.0001",
        "patches": [
            ('weight_decay=0.001)',
             'weight_decay=0.0001)'),
        ],
    },
    # --- Heatmap sigma ---
    {
        "id": "012", "name": "heatmap_sigma=4",
        "patches": [
            ('HEATMAP_SIGMA = 8',
             'HEATMAP_SIGMA = 4'),
        ],
    },
    {
        "id": "013", "name": "heatmap_sigma=12",
        "patches": [
            ('HEATMAP_SIGMA = 8',
             'HEATMAP_SIGMA = 12'),
        ],
    },
    # --- Canny thresholds ---
    {
        "id": "014", "name": "Canny 30/100",
        "patches": [
            ('edges = cv2.Canny(blurred, 50, 150)',
             'edges = cv2.Canny(blurred, 30, 100)'),
        ],
    },
    {
        "id": "015", "name": "Canny 80/200",
        "patches": [
            ('edges = cv2.Canny(blurred, 50, 150)',
             'edges = cv2.Canny(blurred, 80, 200)'),
        ],
    },
    # --- Augmentation ---
    {
        "id": "016", "name": "brightness ±60",
        "patches": [
            ('beta = np.random.uniform(-40, 40)',
             'beta = np.random.uniform(-60, 60)'),
        ],
    },
    {
        "id": "017", "name": "contrast 0.4-1.6",
        "patches": [
            ('alpha = np.random.uniform(0.6, 1.4)',
             'alpha = np.random.uniform(0.4, 1.6)'),
        ],
    },
    {
        "id": "018", "name": "+Gaussian noise",
        "patches": [
            ("""            image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta,
                                0, 255).astype(np.uint8)""",
             """            image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta,
                                0, 255).astype(np.uint8)
            # Gaussian noise
            noise = np.random.normal(0, 10, image_bgr.shape).astype(np.float32)
            image_bgr = np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)"""),
        ],
    },
    # --- ResNet-34 backbone ---
    {
        "id": "019", "name": "ResNet-34 backbone",
        "patches": [
            ('model = models.resnet18(weights=None)',
             'model = models.resnet34(weights=None)'),
            ('cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")',
             'cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth")'),
        ],
    },
    # --- Input size ---
    {
        "id": "020", "name": "IMG_SIZE=448",
        "patches": [
            ('IMG_SIZE = 384', 'IMG_SIZE = 448'),
        ],
        "batch_size": 12,
    },
    {
        "id": "021", "name": "IMG_SIZE=256",
        "patches": [
            ('IMG_SIZE = 384', 'IMG_SIZE = 256'),
        ],
    },
    # --- Head epochs ---
    {
        "id": "022", "name": "head_epochs=5",
        "patches": [
            ('head_epochs = min(3, args.epochs)',
             'head_epochs = min(5, args.epochs)'),
        ],
    },
    # --- Batch size ---
    {
        "id": "023", "name": "batch_size=32",
        "patches": [],
        "batch_size": 32,
    },
    {
        "id": "024", "name": "batch_size=8",
        "patches": [],
        "batch_size": 8,
    },
    # --- Remove Sigmoid, use clamp ---
    {
        "id": "025", "name": "Remove Sigmoid, use clamp",
        "patches": [
            ("""    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 8),
        nn.Sigmoid(),
    )""",
             """    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 8),
    )"""),
            # Clamp outputs in training
            ('outputs = model(images)\n        loss = criterion(outputs, targets)',
             'outputs = model(images).clamp(0, 1)\n        loss = criterion(outputs, targets)'),
            # Clamp outputs in validation
            ('outputs = model(images)\n            loss = criterion(outputs, targets)',
             'outputs = model(images).clamp(0, 1)\n            loss = criterion(outputs, targets)'),
        ],
    },
]


def apply_patches(source, patches):
    """Apply find/replace patches to source string. Returns patched source."""
    result = source
    for old, new in patches:
        if old not in result:
            raise ValueError(f"Patch target not found in source:\n{old[:100]}...")
        result = result.replace(old, new, 1)
    return result


def inject_skip_onnx(source):
    """Insert a return statement before the ONNX export block."""
    marker = "    # Export to ONNX"
    if marker not in source:
        return source
    return source.replace(
        marker,
        "    return  # autoresearch: skip ONNX export\n\n" + marker,
    )


def parse_best_dist(output):
    """Extract 'Best mean corner distance: X.XXXX' from training output."""
    # Match the final summary line
    match = re.search(r"Best mean corner distance:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    # Fallback: look for best val_dist in epoch logs
    best = None
    for m in re.finditer(r"New best! dist=([\d.]+)", output):
        val = float(m.group(1))
        if best is None or val < best:
            best = val
    return best


def load_completed(tsv_path):
    """Load completed experiments from TSV. Returns {id: {status, val_dist, ...}}."""
    completed = {}
    if not tsv_path.exists():
        return completed
    with open(tsv_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                completed[parts[0]] = {
                    "name": parts[1],
                    "status": parts[2],
                    "val_dist": float(parts[3]) if parts[3] != "N/A" else None,
                }
    return completed


def reconstruct_baseline(original_source, experiments, completed):
    """Replay winning patches on original source to get current baseline."""
    source = original_source
    for exp in experiments:
        exp_id = exp["id"]
        if exp_id in completed and completed[exp_id]["status"] == "KEPT":
            try:
                source = apply_patches(source, exp["patches"])
            except ValueError:
                print(f"  Warning: could not replay patches for {exp_id}, "
                      f"continuing with current baseline", flush=True)
    return source


def log_result(tsv_path, exp_id, name, status, val_dist, elapsed, notes=""):
    """Append result to TSV."""
    if not tsv_path.exists():
        with open(tsv_path, "w") as f:
            f.write("id\tname\tstatus\tval_dist\telapsed_s\tnotes\n")
    dist_str = f"{val_dist:.6f}" if val_dist is not None else "N/A"
    with open(tsv_path, "a") as f:
        f.write(f"{exp_id}\t{name}\t{status}\t{dist_str}\t{elapsed:.0f}\t{notes}\n")


def restore_script(backup_path, script_path):
    """Restore training script from backup."""
    if backup_path.exists():
        shutil.copy2(backup_path, script_path)


def run_experiment(exp, baseline_source, epochs=10, default_batch_size=16,
                   timeout=1200, resume_checkpoint=None):
    """
    Run a single experiment:
    1. Apply patches to baseline source
    2. Write modified script
    3. Run training (optionally resuming from checkpoint)
    4. Parse result
    Returns (val_dist, stdout) or (None, error_msg)
    """
    exp_id = exp["id"]
    name = exp["name"]

    # Apply patches
    try:
        patched = apply_patches(baseline_source, exp["patches"])
    except ValueError as e:
        return None, f"PATCH_FAILED: {e}"

    # Skip ONNX export
    patched = inject_skip_onnx(patched)

    # Write patched script
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(patched)

    # Build command
    batch_size = exp.get("batch_size", default_batch_size)
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if resume_checkpoint:
        resume_abs = str(Path(resume_checkpoint).resolve())
        if Path(resume_abs).exists():
            cmd.extend(["--resume", resume_abs])

    print(f"  Running: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BASE_DIR),
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            # Print last few lines for debugging
            lines = output.strip().split("\n")
            tail = "\n".join(lines[-10:])
            print(f"  FAILED (exit {result.returncode}):\n{tail}", flush=True)
            return None, f"EXIT_{result.returncode}: {tail[-200:]}"

        val_dist = parse_best_dist(output)
        if val_dist is None:
            return None, "PARSE_FAILED: could not find val_dist in output"

        return val_dist, output

    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoresearch: autonomous experiment loop")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per experiment (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Default batch size (default: 16)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout per experiment in seconds (default: no timeout)")
    parser.add_argument("--baseline-dist", type=float, default=0.0294,
                        help="Starting best val_dist (default: 0.0294)")
    parser.add_argument("--resume", type=str,
                        default=str(MODELS_DIR / "best_corner_hybrid.pt"),
                        help="Checkpoint to resume from (default: models/best_corner_hybrid.pt)")
    args = parser.parse_args()

    print("=" * 70)
    print("AUTORESEARCH: Autonomous Experiment Loop")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Default batch size: {args.batch_size}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Baseline val_dist: {args.baseline_dist:.4f}")
    print(f"  Resume from: {args.resume}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print("=" * 70, flush=True)

    # Read original source
    original_source = TRAIN_SCRIPT.read_text()

    # Create backup
    shutil.copy2(TRAIN_SCRIPT, BACKUP_FILE)
    print(f"\nBackup saved to {BACKUP_FILE}", flush=True)

    # Register cleanup
    def cleanup():
        restore_script(BACKUP_FILE, TRAIN_SCRIPT)

    atexit.register(cleanup)

    # Load completed experiments
    completed = load_completed(TSV_FILE)
    if completed:
        print(f"\nFound {len(completed)} completed experiments, resuming...", flush=True)
        for eid, info in completed.items():
            print(f"  {eid}: {info['name']} -> {info['status']} "
                  f"(dist={info['val_dist']})", flush=True)

    # Each experiment patches the ORIGINAL source independently.
    # Winning experiments update the resume checkpoint (model weights carry the improvement).

    # Determine current best dist and resume checkpoint
    best_dist = args.baseline_dist
    resume_ckpt = args.resume
    for exp in EXPERIMENTS:
        eid = exp["id"]
        if eid in completed and completed[eid]["status"] == "KEPT":
            if completed[eid]["val_dist"] is not None and completed[eid]["val_dist"] < best_dist:
                best_dist = completed[eid]["val_dist"]
                # Use this experiment's checkpoint if it exists
                exp_ckpt = MODELS_DIR / f"corner_hybrid_exp_{eid}.pt"
                if exp_ckpt.exists():
                    resume_ckpt = str(exp_ckpt)

    print(f"\nCurrent best val_dist: {best_dist:.6f}", flush=True)
    print(f"Resume checkpoint: {resume_ckpt}", flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for exp in EXPERIMENTS:
            exp_id = exp["id"]
            name = exp["name"]

            # Skip completed
            if exp_id in completed:
                print(f"\n[{exp_id}] {name} — SKIPPED (already completed)", flush=True)
                continue

            print(f"\n{'='*70}")
            print(f"[{exp_id}] {name}")
            print(f"  Current best: {best_dist:.6f}")
            print(f"  Resume: {Path(resume_ckpt).name}")
            print(f"  Patches: {len(exp['patches'])}", flush=True)

            start_time = time.time()
            val_dist, output = run_experiment(
                exp, original_source,
                epochs=args.epochs,
                default_batch_size=args.batch_size,
                timeout=args.timeout,
                resume_checkpoint=resume_ckpt,
            )
            elapsed = time.time() - start_time

            if val_dist is None:
                status = "ERROR"
                print(f"  RESULT: ERROR ({output[:100]})", flush=True)
                log_result(TSV_FILE, exp_id, name, status, None, elapsed, output[:200])
            elif val_dist < best_dist:
                status = "KEPT"
                improvement = best_dist - val_dist
                pct = improvement / best_dist * 100
                print(f"  RESULT: KEPT! val_dist={val_dist:.6f} "
                      f"(improved by {improvement:.6f}, {pct:.1f}%)", flush=True)

                # Update best and resume checkpoint
                best_dist = val_dist
                src_ckpt = MODELS_DIR / "best_corner_hybrid.pt"
                dst_ckpt = MODELS_DIR / f"corner_hybrid_exp_{exp_id}.pt"
                if src_ckpt.exists():
                    shutil.copy2(src_ckpt, dst_ckpt)
                    resume_ckpt = str(dst_ckpt)
                    print(f"  Checkpoint saved: {dst_ckpt.name}", flush=True)

                log_result(TSV_FILE, exp_id, name, status, val_dist, elapsed)
            else:
                status = "DISCARDED"
                diff = val_dist - best_dist
                print(f"  RESULT: DISCARDED val_dist={val_dist:.6f} "
                      f"(worse by {diff:.6f})", flush=True)
                log_result(TSV_FILE, exp_id, name, status, val_dist, elapsed)

            # Always restore original before next experiment
            restore_script(BACKUP_FILE, TRAIN_SCRIPT)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!", flush=True)
    finally:
        # Restore original script
        restore_script(BACKUP_FILE, TRAIN_SCRIPT)
        print(f"\nScript restored from backup.", flush=True)

    # Print summary
    print(f"\n{'='*70}")
    print("AUTORESEARCH SUMMARY")
    print(f"{'='*70}")
    completed = load_completed(TSV_FILE)
    kept = [e for e in completed.values() if e["status"] == "KEPT"]
    discarded = [e for e in completed.values() if e["status"] == "DISCARDED"]
    errors = [e for e in completed.values() if e["status"] == "ERROR"]
    print(f"  Total: {len(completed)} | Kept: {len(kept)} | "
          f"Discarded: {len(discarded)} | Errors: {len(errors)}")
    print(f"  Best val_dist: {best_dist:.6f}")
    if kept:
        print("\n  Winning experiments:")
        for eid, info in completed.items():
            if info["status"] == "KEPT":
                print(f"    {eid}: {info['name']} -> {info['val_dist']:.6f}")
    print(f"\nResults: {TSV_FILE}", flush=True)


if __name__ == "__main__":
    main()
