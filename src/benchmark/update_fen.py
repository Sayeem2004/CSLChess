import chess
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR


def update_fen(phase: str):
    path = os.path.join(DATA_DIR, phase, "puzzles.csv")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FEN", "FirstMove", "UpdatedFEN"])
        writer.writeheader()
        for row in rows:
            board = chess.Board(row["FEN"])
            board.push(chess.Move.from_uci(row["FirstMove"]))
            writer.writerow({"FEN": row["FEN"], "FirstMove": row["FirstMove"], "UpdatedFEN": board.fen()})

    print(f"[update_fen] {phase}: wrote {path}")


if __name__ == "__main__":
    for phase in PHASES:
        update_fen(phase)
