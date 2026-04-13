"""
Plays a chess game where our C++ engine (alpha-beta or monte-carlo skeleton)
faces Stockfish.  The chosen C++ source is compiled automatically on startup.

Usage:
    python main.py [--algorithm {alpha-beta, monte-carlo}]
                   [--color {white, black}]
                   [--timelimit N]
"""
from stockfish import Stockfish
import subprocess
import argparse
import ctypes
import chess
import sys
import os

from utils import print_green, print_yellow, print_red


SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
STANDARD_DIR = os.path.join(SRC_DIR, "standard")
MOVE_BUF_LEN = 8  # longest UCI move is 5 chars (e.g. "e7e8q\0")


def compile_and_load(algorithm: str):
    """Compile the C++ shared library for `algorithm` and return its ctypes fn."""
    if algorithm == "alpha-beta":
        src     = os.path.join(STANDARD_DIR, "alpha-beta.cpp")
        lib_out = os.path.join(STANDARD_DIR, "alpha-beta.so")
        fn_name = "best_move_alpha_beta"
    else:
        src     = os.path.join(STANDARD_DIR, "monte-carlo.cpp")
        lib_out = os.path.join(STANDARD_DIR, "monte-carlo.so")
        fn_name = "best_move_monte_carlo"

    print_yellow(f"[build] compiling {os.path.basename(src)} ...")
    result = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-shared", "-fPIC", src, "-o", lib_out],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print_red(f"[build] failed: \n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print_green(f"[build] succeeded at {os.path.relpath(lib_out)}\n")

    lib = ctypes.CDLL(lib_out)
    fn  = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_char_p,  # fen
        ctypes.c_int,     # depth / num_simulations
        ctypes.c_char_p,  # out_move buffer
        ctypes.c_int,     # out_len
    ]
    fn.restype = ctypes.c_int
    return fn


def cpp_best_move(fn, board: chess.Board, param: int) -> chess.Move | None:
    """
    Invoke the C++ engine for the current position.
    Returns a legal chess.Move, or None if the engine returned an error / illegal move.
    """
    fen_bytes = board.fen().encode()
    buf       = ctypes.create_string_buffer(MOVE_BUF_LEN)
    rc        = fn(fen_bytes, param, buf, MOVE_BUF_LEN)

    if rc != 0: return None
    uci_str = buf.value.decode().strip()
    try: move = chess.Move.from_uci(uci_str)
    except chess.InvalidMoveError: return None
    return move if move in board.legal_moves else None


def play_game(algorithm: str, our_color: chess.Color, depth: int, stockfish_path: str, stockfish_depth: int) -> None:
    fn        = compile_and_load(algorithm)
    board     = chess.Board()
    color_str = "White" if our_color == chess.WHITE else "Black"

    print_yellow(f"{'='*60}")
    print_yellow(f"  Engine  : {algorithm} ({color_str})")
    print_yellow(f"  Opponent: Stockfish (depth {stockfish_depth})")
    print_yellow(f"{'='*60}\n")

    stockfish   = Stockfish(path=stockfish_path, depth=stockfish_depth)

    move_number = 1
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print(f"--- Move {move_number} (White) ---")
        else:
            print(f"--- Move {move_number} (Black) ---")
            move_number += 1
        print(board)
        print()

        if board.turn == our_color:
            move = cpp_best_move(fn, board, depth)
            if move is None:
                move = next(iter(board.legal_moves))
                print_red(f"  [{algorithm}] (fallback)  {board.san(move)}\n")
            else:
                print_green(f"  [{algorithm}]  {board.san(move)}\n")

        else:
            stockfish.set_fen_position(board.fen())
            best_uci = stockfish.get_best_move()
            move     = chess.Move.from_uci(best_uci)
            print_yellow(f"  [stockfish]  {board.san(move)}\n")

        board.push(move)

    print("="*60)
    print(board)
    outcome = board.outcome()

    if outcome is None or outcome.winner is None: print("\n  Result: draw")
    elif outcome.winner == our_color: print(f"\n  Result: {algorithm} wins!")
    else: print("\n  Result: Stockfish wins.")
    print("="*60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard CPP Engine vs Stockfish")
    parser.add_argument("--algorithm", choices=["alpha-beta", "monte-carlo"], default="alpha-beta", help="Search algorithm (default: alpha-beta)")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Color our engine plays (default: white)")
    parser.add_argument("--depth", type=int, default=5, help="Search depth / simulation count for our engine (default: 5)")
    parser.add_argument("--stockfish", default="stockfish", help="Path to the Stockfish binary (default: stockfish)")
    parser.add_argument("--stockfish-depth", type=int, default=5, help="Stockfish search depth (default: 5)")

    args      = parser.parse_args()
    our_color = chess.WHITE if args.color == "white" else chess.BLACK
    play_game(args.algorithm, our_color, args.depth, args.stockfish, args.stockfish_depth)
