import chess

from utils.random import print_green, print_yellow, print_red
from utils.move_depth import cpp_best_move_depth, stockfish_best_move_depth, csl_best_move_depth
from utils.move_time  import cpp_best_move_time,  stockfish_best_move_time,  csl_best_move_time
from utils.move_flops import cpp_best_move_flops, stockfish_best_move_flops, csl_best_move_flops


def engine_best_move(algorithm: str, engine, board: chess.Board, budget_type: str, budget: int) -> chess.Move | None:
    """Dispatch to the right budget-type move function for our engine."""
    is_csl = algorithm.startswith("csl-")
    if budget_type == "depth":
        return csl_best_move_depth(engine, board, budget) if is_csl else cpp_best_move_depth(engine, board, budget)
    if budget_type == "time":
        return csl_best_move_time(engine, board, budget)  if is_csl else cpp_best_move_time(engine, board, budget)
    if budget_type == "flops":
        return csl_best_move_flops(engine, board, budget) if is_csl else cpp_best_move_flops(engine, board, budget)


def stockfish_best_move(stockfish, board: chess.Board, budget_type: str, budget: int) -> chess.Move | None:
    """Dispatch to the right budget-type move function for Stockfish."""
    if budget_type == "depth": return stockfish_best_move_depth(stockfish, board, budget)
    if budget_type == "time":  return stockfish_best_move_time(stockfish, board, budget)
    if budget_type == "flops": return stockfish_best_move_flops(stockfish, board, budget)
