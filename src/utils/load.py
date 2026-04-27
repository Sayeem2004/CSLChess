import subprocess
import platform
import ctypes
import glob
import sys
import os

from utils.random import print_green, print_yellow, print_red


SRC_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDARD_DIR       = os.path.join(SRC_DIR, "standard")
STOCKFISH_DIR      = os.path.join(SRC_DIR, "..", "stockfish")
STOCKFISH_SRC_DIR  = os.path.join(STOCKFISH_DIR, "stockfish-11-src")
STOCKFISH_UNIX_BIN = os.path.join(STOCKFISH_DIR, "stockfish-11-modern")
STOCKFISH_WIN_EXE  = os.path.join(STOCKFISH_DIR, "stockfish-11-modern.exe")


MOVE_BUF_LEN = 8 # Longest UCI move is 5 chars (e.g. "e7e8q\0")


def _compile_library(src: str, lib_out: str) -> ctypes.CDLL:
    """Compile src + evaluate.cpp + stockfish code into a shared library and return the CDLL."""
    if os.path.exists(lib_out):
        print_yellow(f"[build] {os.path.basename(lib_out)} already exists, skipping compilation\n")
    else:
        evaluate_src = os.path.join(STANDARD_DIR, "evaluate.cpp")
        sf_sources   = glob.glob(os.path.join(STOCKFISH_SRC_DIR, "*.cpp")) + \
                       glob.glob(os.path.join(STOCKFISH_SRC_DIR, "syzygy", "*.cpp"))
        sf_sources   = [f for f in sf_sources if os.path.basename(f) != "main.cpp"]

        print_yellow(f"[build] compiling {os.path.basename(src)} ...")
        cmd = ["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
               src, evaluate_src, *sf_sources, "-o", lib_out]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print_red(f"[build] failed:\n{result.stderr}")
            sys.exit(1)
        print_green(f"[build] succeeded at {os.path.relpath(lib_out)}\n")

    lib_path = os.path.abspath(lib_out)
    return ctypes.CDLL(lib_path, winmode=0) if hasattr(os, "add_dll_directory") else ctypes.CDLL(lib_path)


def _bind_fn(lib: ctypes.CDLL, fn_name: str):
    """Extract a best-move function from a loaded library and attach its ctypes signature."""
    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_char_p,  # fen
        ctypes.c_int,     # budget (depth / time_ms / flop_budget)
        ctypes.c_char_p,  # out_move buffer
        ctypes.c_int,     # out_len
    ]
    fn.restype = ctypes.c_int
    return fn


def load_standard_alpha_beta() -> dict:
    """Compile and load the standard C++ alpha-beta engine.
    Returns a dict with keys 'depth', 'time', 'flops' mapping to their ctypes fns."""
    src     = os.path.join(STANDARD_DIR, "alpha-beta.cpp")
    lib_out = os.path.join(STANDARD_DIR, "alpha-beta.so")
    lib = _compile_library(src, lib_out)
    return {
        "depth": _bind_fn(lib, "best_move_alpha_beta_depth"),
        "time":  _bind_fn(lib, "best_move_alpha_beta_time"),
        "flops": _bind_fn(lib, "best_move_alpha_beta_flops"),
    }


def load_standard_monte_carlo() -> dict:
    """Compile and load the standard C++ MCTS engine.
    Returns a dict with keys 'depth', 'time', 'flops' mapping to their ctypes fns."""
    src     = os.path.join(STANDARD_DIR, "monte-carlo.cpp")
    lib_out = os.path.join(STANDARD_DIR, "monte-carlo.so")
    lib = _compile_library(src, lib_out)
    return {
        "depth": _bind_fn(lib, "best_move_monte_carlo_depth"),
        "time":  _bind_fn(lib, "best_move_monte_carlo_time"),
        "flops": _bind_fn(lib, "best_move_monte_carlo_flops"),
    }


def load_csl_alpha_beta() -> dict:
    """Load the CSL alpha-beta engine. Not yet implemented."""
    pass


def load_csl_monte_carlo() -> dict:
    """Load the CSL MCTS engine. Not yet implemented."""
    pass


def load_stockfish_unix(path: str = STOCKFISH_UNIX_BIN):
    """Return a Stockfish instance backed by the Unix binary at `path`."""
    from stockfish import Stockfish
    # Crippled settings for testing against a much weaker opponent (~1100 ELO)
    return Stockfish(path=path, parameters={"Skill Level": 0, "UCI_LimitStrength": True, "UCI_Elo": 1320})


def load_stockfish_windows(path: str = STOCKFISH_WIN_EXE):
    """Return a Stockfish instance backed by the Windows binary at `path`."""
    from stockfish import Stockfish
    # Crippled settings for testing against a much weaker opponent (~1100 ELO)
    return Stockfish(path=path, parameters={"Skill Level": 0, "UCI_LimitStrength": True, "UCI_Elo": 1320})


def load_engine(algorithm: str) -> dict:
    """Return a dict of {budget_type: ctypes_fn} for the chosen algorithm."""
    if algorithm == "cpp-alpha-beta":  return load_standard_alpha_beta()
    if algorithm == "cpp-monte-carlo": return load_standard_monte_carlo()
    if algorithm == "csl-alpha-beta":  return load_csl_alpha_beta()
    if algorithm == "csl-monte-carlo": return load_csl_monte_carlo()


def load_stockfish():
    """Load Stockfish using the platform-appropriate binary."""
    if platform.system() == "Windows":
        return load_stockfish_windows()
    return load_stockfish_unix()  # Darwin + Linux