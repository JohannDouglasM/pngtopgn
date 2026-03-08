"""
Chess board recognition model.

End-to-end: full board image → 64 squares × 13 classes.
Architecture: EfficientNetV2-S backbone → Linear head.
Based on Fenify-3D approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 13 classes per square
# 0 = empty
# 1-6 = white pawn, knight, bishop, rook, queen, king
# 7-12 = black pawn, knight, bishop, rook, queen, king
NUM_CLASSES = 13
NUM_SQUARES = 64

# Class weights to handle imbalance (most squares are empty)
CLASS_WEIGHTS = torch.tensor([
    64/32,   # empty (32 out of 64 in starting pos)
    64/8,    # white pawn
    64/2,    # white knight
    64/2,    # white bishop
    64/2,    # white rook
    64/1,    # white queen
    64/1,    # white king
    64/8,    # black pawn
    64/2,    # black knight
    64/2,    # black bishop
    64/2,    # black rook
    64/1,    # black queen
    64/1,    # black king
])

# Mapping from FEN chars to class indices
FEN_TO_CLASS = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}

CLASS_TO_FEN = {v: k for k, v in FEN_TO_CLASS.items()}
CLASS_TO_FEN[0] = None  # empty


def fen_to_target(fen: str) -> torch.Tensor:
    """Convert FEN piece placement string to a (64,) tensor of class indices.

    Square ordering: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
    (rank-major, matching python-chess convention)
    """
    target = torch.zeros(64, dtype=torch.long)
    ranks = fen.split("/")
    for rank_idx, rank_str in enumerate(ranks):
        # FEN rank 8 (index 0) = chess rank 8 = row index 7
        chess_rank = 7 - rank_idx
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                sq = chess_rank * 8 + file_idx
                target[sq] = FEN_TO_CLASS.get(ch, 0)
                file_idx += 1
    return target


def target_to_fen(target: torch.Tensor) -> str:
    """Convert a (64,) tensor of class indices back to FEN piece placement."""
    ranks = []
    for chess_rank in range(7, -1, -1):  # rank 8 down to rank 1
        rank_str = ""
        empty = 0
        for file_idx in range(8):
            sq = chess_rank * 8 + file_idx
            cls = target[sq].item()
            fen_ch = CLASS_TO_FEN.get(cls)
            if fen_ch is None:
                empty += 1
            else:
                if empty > 0:
                    rank_str += str(empty)
                    empty = 0
                rank_str += fen_ch
        if empty > 0:
            rank_str += str(empty)
        ranks.append(rank_str)
    return "/".join(ranks)


class ChessRecognitionModel(nn.Module):
    """End-to-end chess board recognition model.

    Input: (B, 3, 400, 400) RGB image
    Output: (B, 64, 13) logits per square
    """

    def __init__(self, backbone: str = "efficientnet_v2_s", pretrained: bool = True):
        super().__init__()

        if backbone == "efficientnet_v2_s":
            # Load without hash check — torchvision's check_hash fails on some systems
            self.backbone = models.efficientnet_v2_s(weights=None)
            if pretrained:
                import os
                cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth")
                if os.path.exists(cached):
                    state = torch.load(cached, map_location="cpu", weights_only=True)
                    self.backbone.load_state_dict(state)
                else:
                    print("Warning: EfficientNet pretrained weights not found, using random init")
            backbone_out = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(backbone_out, NUM_SQUARES * NUM_CLASSES),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        logits = logits.view(-1, NUM_SQUARES, NUM_CLASSES)
        return logits


def load_fenify_weights(model: ChessRecognitionModel, jit_path: str):
    """Load weights from a Fenify-3D TorchScript model."""
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    jit_state = jit_model.state_dict()

    model_state = model.state_dict()
    loaded = 0
    skipped = 0

    # Fenify uses "resnet." prefix for backbone, "outputs." for head
    # Our model uses "backbone." and "head.2."
    for fenify_key, val in jit_state.items():
        our_key = None
        if fenify_key.startswith("resnet."):
            our_key = fenify_key.replace("resnet.", "backbone.", 1)
        elif fenify_key == "outputs.weight":
            our_key = "head.2.weight"
        elif fenify_key == "outputs.bias":
            our_key = "head.2.bias"

        if our_key and our_key in model_state and model_state[our_key].shape == val.shape:
            model_state[our_key] = val
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state)
    print(f"Loaded {loaded}/{len(jit_state)} params from Fenify weights (skipped {skipped})")
    return model
