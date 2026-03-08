#!/usr/bin/env npx tsx
/**
 * Test chess board recognition via Anthropic API on a local image.
 *
 * Usage:
 *   npx tsx scripts/test_local.ts <IMAGE_PATH>
 *
 * Requires ANTHROPIC_API_KEY environment variable.
 */

import * as fs from "fs";
import * as path from "path";

const IMAGE_PATH = process.argv[2];
if (!IMAGE_PATH) {
  console.error("Usage: npx tsx scripts/test_local.ts <IMAGE_PATH>");
  process.exit(1);
}

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) {
  console.error("Error: ANTHROPIC_API_KEY environment variable is required.");
  process.exit(1);
}

const CLAUDE_MODEL = "claude-opus-4-20250514";

const SYSTEM_PROMPT = `You are a chess position recognition expert. You will be shown a photograph of a chess board. Your task is to identify every piece on the board and return the position as a FEN piece placement string (only the first field — no side to move, castling, etc.).

You MUST use chain-of-thought analysis. Go through the board systematically:
1. First, determine the board orientation (which side is white, which is black). White pieces are typically lighter colored, black pieces darker. Assume white plays from the bottom unless clearly otherwise.
2. Then analyze EACH rank from rank 8 (top) to rank 1 (bottom), and for each rank go file by file from a to h.
3. For each square, state whether it is empty or occupied, and if occupied, identify the piece type and color.
4. After analyzing all 64 squares, construct the FEN piece placement string.

Rules:
- Use standard FEN notation: uppercase for white (KQRBNP), lowercase for black (kqrbnp), numbers for consecutive empty squares, "/" between ranks.
- Rank 8 (black's back rank) comes first, rank 1 (white's back rank) comes last.
- If you cannot identify a piece with certainty, make your best guess.
- After your analysis, output the final FEN on its own line prefixed with "FEN: "

Example final line:
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR`;

const PIECE_UNICODE: Record<string, string> = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

function parseFenFromResponse(text: string): string | null {
  const lines = text.trim().split("\n");
  for (const line of lines) {
    let trimmed = line.trim();
    // Strip "FEN: " or "FEN:" prefix
    if (trimmed.toUpperCase().startsWith("FEN:")) {
      trimmed = trimmed.slice(4).trim();
    }
    // Remove backticks
    trimmed = trimmed.replace(/`/g, "");
    const candidate = trimmed.split(" ")[0];
    if (/^[rnbqkpRNBQKP1-8/]+$/.test(candidate)) {
      const ranks = candidate.split("/");
      if (ranks.length === 8) {
        const valid = ranks.every((rank) => {
          let count = 0;
          for (const ch of rank) {
            count += ch >= "1" && ch <= "8" ? parseInt(ch) : 1;
          }
          return count === 8;
        });
        if (valid) return candidate;
      }
    }
  }
  return null;
}

async function main() {
  const absPath = path.resolve(IMAGE_PATH);
  if (!fs.existsSync(absPath)) {
    console.error(`File not found: ${absPath}`);
    process.exit(1);
  }

  const imageBuffer = fs.readFileSync(absPath);
  const base64 = imageBuffer.toString("base64");

  const ext = path.extname(absPath).toLowerCase();
  let mediaType = "image/jpeg";
  if (ext === ".png") mediaType = "image/png";
  else if (ext === ".gif") mediaType = "image/gif";
  else if (ext === ".webp") mediaType = "image/webp";

  console.log(`Image: ${absPath}`);
  console.log(`Size: ${(imageBuffer.length / 1024).toFixed(0)} KB`);
  console.log(`Sending to Claude (${CLAUDE_MODEL})...\n`);

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: CLAUDE_MODEL,
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: base64,
              },
            },
            {
              type: "text",
              text: "What is the chess position in this image? Return only the FEN piece placement string.",
            },
          ],
        },
      ],
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`API error (${response.status}): ${errorBody}`);
    process.exit(1);
  }

  const data = await response.json();
  const text: string = data.content?.[0]?.text ?? "";
  console.log(`Raw response: ${text}\n`);

  const fen = parseFenFromResponse(text);
  if (!fen) {
    console.error(`Could not parse FEN from response: "${text}"`);
    process.exit(1);
  }

  // Print board
  const ranks = fen.split("/");
  console.log("Board:");
  console.log("  a b c d e f g h");
  for (let row = 0; row < 8; row++) {
    let line = `${8 - row} `;
    let col = 0;
    for (const ch of ranks[row]) {
      if (ch >= "1" && ch <= "8") {
        for (let i = 0; i < parseInt(ch); i++) {
          line += "· ";
          col++;
        }
      } else {
        line += (PIECE_UNICODE[ch] ?? ch) + " ";
        col++;
      }
    }
    console.log(line);
  }

  const fullFen = fen + " w KQkq - 0 1";
  console.log(`\nFEN: ${fen}`);
  console.log(`Full: ${fullFen}`);
  console.log(`Lichess: https://lichess.org/analysis/${encodeURIComponent(fullFen)}`);
}

main().catch(console.error);
