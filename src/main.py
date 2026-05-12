"""
Plays one or more chess games where our engine faces Stockfish.
The chosen algorithm is compiled/loaded automatically on startup.

Usage:
    python main.py --algorithm {cpp-alpha-beta,cpp-monte-carlo}
                   (--depth DEPTH | --time TIME_MS | --cycles CYCLES)
                   [--color {white,black}] [--num NUM] [--verbose]
"""
import argparse
import random
import chess

from utils import print_green, print_yellow, print_red, engine_best_move, stockfish_best_move
from utils.load import load_engine, load_stockfish, SF_ELO_MIN, SF_ELO_MAX


def play_game(algorithm: str, our_color: chess.Color, budget_type: str, budget: int,
              engine, stockfish, verbose: bool) -> str:
    """
    Play a single game. Returns 'win', 'loss', or 'draw' from our engine's perspective.
    `engine` and `stockfish` are pre-loaded handles (compiled/loaded once before the game loop).
    """
    board     = chess.Board()
    color_str = "White" if our_color == chess.WHITE else "Black"

    if verbose:
        print_yellow(f"{'='*60}")
        print_yellow(f"  Engine  : {algorithm} ({color_str})")
        print_yellow(f"  Budget  : {budget_type}={budget}")
        print_yellow(f"  Opponent: Stockfish")
        print_yellow(f"{'='*60}\n")

    move_number = 1
    while not board.is_game_over():
        if verbose:
            side = "White" if board.turn == chess.WHITE else "Black"
            print(f"--- Move {move_number} ({side}) ---")
            print(board)
            print()

        if board.turn == our_color:
            move = engine_best_move(algorithm, engine[budget_type], board, budget_type, budget)
            if move is None:
                move = next(iter(board.legal_moves))
                if verbose: print_red(f"  [{algorithm}] (fallback)  {board.san(move)}\n")
            else:
                if verbose: print_green(f"  [{algorithm}]  {board.san(move)}\n")
        else:
            move = stockfish_best_move(stockfish, board, budget_type, budget)
            if move is None:
                move = next(iter(board.legal_moves))
                if verbose: print_red(f"  [stockfish] (fallback)  {board.san(move)}\n")
            else:
                if verbose: print_yellow(f"  [stockfish]  {board.san(move)}\n")

        if board.turn == chess.BLACK:
            move_number += 1
        board.push(move)

    outcome = board.outcome()
    if verbose:
        print("="*60)
        print(board)
        print("="*60 + "\n")

    if outcome is None or outcome.winner is None: return "draw"
    return "win" if outcome.winner == our_color else "loss"


if __name__ == "__main__":
    algorithms = ["cpp-alpha-beta", "cpp-monte-carlo", "cpp-monte-carlo-rp"]
    parser = argparse.ArgumentParser(description="Our Chess Engine vs Stockfish")
    parser.add_argument("--algorithm", required=True, choices=algorithms,
                        help="Search algorithm to use")
    parser.add_argument("--color", choices=["white", "black"], default=None,
                        help="Color our engine plays (default: random per game)")
    parser.add_argument("--num", type=int, default=1,
                        help="Number of games to play (default: 1)")
    parser.add_argument("--elo", type=int, default=SF_ELO_MIN, metavar=f"[{SF_ELO_MIN}-{SF_ELO_MAX}]",
                        help=f"Stockfish target ELO via UCI_LimitStrength (default: {SF_ELO_MIN})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print board and move output each turn")

    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--depth",  type=int, metavar="DEPTH",   help="Limit search by depth (alpha-beta, stockfish) or simulation count (MCTS)")
    budget.add_argument("--time",   type=int, metavar="TIME_MS", help="Limit search by time in milliseconds")
    budget.add_argument("--cycles", type=int, metavar="CYCLES",  help="Limit search by cycle count")
    args = parser.parse_args()

    if args.depth is not None:  budget_type, budget_val = "depth", args.depth
    elif args.time is not None: budget_type, budget_val = "time",  args.time
    else:                       budget_type, budget_val = "cycles", args.cycles

    engine    = load_engine(args.algorithm)
    stockfish = load_stockfish(args.elo)
    wins, losses, draws = 0, 0, 0

    for i in range(args.num):
        if args.color is None: our_color = random.choice([chess.WHITE, chess.BLACK])
        else: our_color = chess.WHITE if args.color == "white" else chess.BLACK
        color_str = "white" if our_color == chess.WHITE else "black"
        print_yellow(f"Game {i + 1}/{args.num}  (our engine plays {color_str})")

        result = play_game(args.algorithm, our_color, budget_type, budget_val, engine, stockfish, args.verbose)
        if result == "win":    wins   += 1; print_green(f"  -> win\n")
        elif result == "loss": losses += 1; print_red(f"  -> loss\n")
        else:                  draws  += 1; print_yellow(f"  -> draw\n")

    print_yellow(f"{'='*60}")
    print_yellow(f"  Final Results")
    print_yellow(f"{'='*60}")
    print_yellow(f"  Algorithm : {args.algorithm}")
    print_yellow(f"  Budget    : {budget_type}={budget_val}")
    print_yellow(f"  Games     : {args.num}")
    print_yellow(f"  Opponent  : Stockfish {args.elo} ELO")
    print_yellow(f"  Color     : {args.color or 'random'}")
    print_yellow(f"{'='*60}")
    print_green( f"  Wins  : {wins}")
    print_red(   f"  Losses: {losses}")
    print_yellow(f"  Draws : {draws}")
    print_yellow(f"  Win % : {100 * wins / args.num:.1f}%")
    print_yellow(f"{'='*60}")
