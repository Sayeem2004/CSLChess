import chess


def cpp_best_move_flops(fn, board: chess.Board, flop_budget: int) -> chess.Move | None:
    """
    Query the C++ engine (alpha-beta or MCTS) for the best move within a FLOP budget.
    `fn`          - ctypes function returned by load_standard_alpha_beta / load_standard_monte_carlo.
    `flop_budget` - floating-point operation budget.
    Not yet implemented.
    """
    pass


def stockfish_best_move_flops(stockfish, board: chess.Board, flop_budget: int) -> chess.Move | None:
    """
    Query a Stockfish instance for the best move within a FLOP budget.
    `stockfish`   - Stockfish instance returned by load_stockfish_unix / load_stockfish_windows.
    `flop_budget` - floating-point operation budget.
    Not yet implemented.
    """
    pass


def csl_best_move_flops(fn, board: chess.Board, flop_budget: int) -> chess.Move | None:
    """
    Query the CSL engine for the best move within a FLOP budget.
    `fn`          - CSL engine handle returned by load_csl_alpha_beta / load_csl_monte_carlo.
    `flop_budget` - floating-point operation budget.
    Not yet implemented.
    """
    pass