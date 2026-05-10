import ctypes
import chess

from utils.load import MOVE_BUF_LEN


def cpp_best_move_cycles(fn, board: chess.Board, megacycle_budget: int) -> chess.Move | None:
    """
    Query the C++ engine (alpha-beta or MCTS) for the best move within a cycle budget.
    `fn`              - ctypes function returned by load_standard_alpha_beta / load_standard_monte_carlo.
    `megacycle_budget` - cycle budget in millions (scaled to full cycles inside C++).
    Returns a legal chess.Move, or None on error / illegal result.
    """
    fen_bytes = board.fen().encode()
    buf       = ctypes.create_string_buffer(MOVE_BUF_LEN)
    rc        = fn(fen_bytes, megacycle_budget, buf, MOVE_BUF_LEN)

    if rc < 0: return None
    uci_str = buf.value.decode().strip()
    try: move = chess.Move.from_uci(uci_str)
    except chess.InvalidMoveError: return None
    return move if move in board.legal_moves else None


STOCKFISH_CYCLES_PER_NODE = 1800 # Obtained from benchmark.calibrate-stockfish.py


def stockfish_best_move_cycles(stockfish, board: chess.Board, megacycle_budget: int) -> chess.Move | None:
    """
    Query a Stockfish instance for the best move within a cycle budget.
    `stockfish`       - Stockfish instance returned by load_stockfish_unix / load_stockfish_windows.
    `megacycle_budget` - cycle budget in millions, converted to node count via STOCKFISH_CYCLES_PER_NODE.
    Returns a legal chess.Move, or None on error / illegal result.
    """
    node_budget = max(1, (megacycle_budget * 1_000_000) // STOCKFISH_CYCLES_PER_NODE)
    stockfish.set_fen_position(board.fen())
    stockfish._put(f"go nodes {node_budget}")

    best_uci = None
    while True:
        line = stockfish._read_line()
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] != "(none)":
                best_uci = parts[1]
            break

    if best_uci is None: return None
    try: move = chess.Move.from_uci(best_uci)
    except chess.InvalidMoveError: return None
    return move if move in board.legal_moves else None
