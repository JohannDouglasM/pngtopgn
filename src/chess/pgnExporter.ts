/**
 * Convert a FEN string into PGN format for export.
 */

export function fenToPgn(
  fen: string,
  options?: {
    white?: string;
    black?: string;
    event?: string;
    date?: string;
    site?: string;
  }
): string {
  const today = new Date();
  const dateStr =
    options?.date ??
    `${today.getFullYear()}.${String(today.getMonth() + 1).padStart(2, "0")}.${String(today.getDate()).padStart(2, "0")}`;

  const headers = [
    `[Event "${options?.event ?? "Scanned Position"}"]`,
    `[Site "${options?.site ?? "pngtopgn App"}"]`,
    `[Date "${dateStr}"]`,
    `[Round "?"]`,
    `[White "${options?.white ?? "?"}"]`,
    `[Black "${options?.black ?? "?"}"]`,
    `[Result "*"]`,
    `[FEN "${fen}"]`,
    `[SetUp "1"]`,
  ];

  return headers.join("\n") + "\n\n*\n";
}

/**
 * Generate a Lichess analysis URL from a FEN string.
 */
export function lichessAnalysisUrl(fen: string): string {
  return `https://lichess.org/analysis/${encodeURIComponent(fen)}`;
}
