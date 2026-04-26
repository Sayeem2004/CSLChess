import ctypes
import chess

from utils.load import MOVE_BUF_LEN


def cpp_best_move_depth(fn, board: chess.Board, depth: int) -> chess.Move | None:
    """
    Query the C++ engine (alpha-beta or MCTS) for the best move with a depth limit.
    `fn`    - ctypes function returned by load_standard_alpha_beta / load_standard_monte_carlo.
    `depth` - search depth directly (alpha-beta) or exponentiated to get rough simulation count (MCTS).
    Returns a legal chess.Move, or None on error / illegal result.
    """
    fen_bytes = board.fen().encode()
    buf       = ctypes.create_string_buffer(MOVE_BUF_LEN)
    rc        = fn(fen_bytes, depth, buf, MOVE_BUF_LEN)

    if rc != 0: return None
    uci_str = buf.value.decode().strip()
    try: move = chess.Move.from_uci(uci_str)
    except chess.InvalidMoveError: return None
    return move if move in board.legal_moves else None


def stockfish_best_move_depth(stockfish, board: chess.Board, depth: int) -> chess.Move | None:
    """
    Query a Stockfish instance for the best move with a depth limit.
    `stockfish` - Stockfish instance returned by load_stockfish_unix / load_stockfish_windows.
    `depth`     - search depth passed to Stockfish.
    Returns a legal chess.Move, or None if Stockfish returns nothing.
    """
    stockfish.set_depth(depth)
    stockfish.set_fen_position(board.fen())
    best_uci = stockfish.get_best_move()

    if best_uci is None: return None
    try: move = chess.Move.from_uci(best_uci)
    except chess.InvalidMoveError: return None
    return move if move in board.legal_moves else None


def csl_best_move_depth(fn, board: chess.Board, depth: int) -> chess.Move | None:
    """
    Query the CSL engine for the best move with a depth limit.
    Not yet implemented.
    """
    pass
