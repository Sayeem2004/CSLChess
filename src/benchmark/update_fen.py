import chess
import csv
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PHASES   = ["early", "mid", "late"]


def update_fen(phase: str):
    in_path  = os.path.join(DATA_DIR, phase, "puzzles.csv")
    out_path = os.path.join(DATA_DIR, phase, "updatedFEN.csv")

    with open(in_path, newline="") as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["UpdatedFEN"])
        writer.writeheader()

        for row in reader:
            board = chess.Board(row["FEN"])
            move  = chess.Move.from_uci(row["FirstMove"])
            board.push(move)
            writer.writerow({ "UpdatedFEN": board.fen() })

    print(f"[update_fen] {phase}: wrote {out_path}")


if __name__ == "__main__":
    for phase in PHASES:
        update_fen(phase)
