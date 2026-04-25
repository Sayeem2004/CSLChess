import chess


def cpp_best_move_time(fn, board: chess.Board, time_ms: int) -> chess.Move | None:
    """
    Query the C++ engine (alpha-beta or MCTS) for the best move within a time limit.
    `fn`      - ctypes function returned by load_standard_alpha_beta / load_standard_monte_carlo.
    `time_ms` - time budget in milliseconds.
    Not yet implemented.
    """
    pass


def stockfish_best_move_time(stockfish, board: chess.Board, time_ms: int) -> chess.Move | None:
    """
    Query a Stockfish instance for the best move within a time limit.
    `stockfish` - Stockfish instance returned by load_stockfish_unix / load_stockfish_windows.
    `time_ms`   - time budget in milliseconds.
    Not yet implemented.
    """
    pass


def csl_best_move_time(fn, board: chess.Board, time_ms: int) -> chess.Move | None:
    """
    Query the CSL engine for the best move within a time limit.
    `fn`      - CSL engine handle returned by load_csl_alpha_beta / load_csl_monte_carlo.
    `time_ms` - time budget in milliseconds.
    Not yet implemented.
    """
    pass