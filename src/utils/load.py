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


def _compile_and_load(src: str, lib_out: str, fn_name: str):
    """Compile src + utils.cpp + Stockfish sources into a shared library and return the ctypes fn."""
    utils_src  = os.path.join(STANDARD_DIR, "utils.cpp")
    sf_sources = glob.glob(os.path.join(STOCKFISH_SRC_DIR, "*.cpp")) + \
                 glob.glob(os.path.join(STOCKFISH_SRC_DIR, "syzygy", "*.cpp"))
    sf_sources = [f for f in sf_sources if os.path.basename(f) != "main.cpp"]

    if os.path.exists(lib_out):
        print_yellow(f"[build] {os.path.basename(lib_out)} already exists, skipping compilation\n")
    else:
        print_yellow(f"[build] compiling {os.path.basename(src)} ...")
        cmd = ["g++", "-O2", "-std=c++17", "-shared", "-fPIC"]
        cmd.extend([src, utils_src])
        cmd.extend(sf_sources)
        cmd.extend(["-o", lib_out])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print_red(f"[build] failed:\n{result.stderr}")
            sys.exit(1)
        print_green(f"[build] succeeded at {os.path.relpath(lib_out)}\n")

    lib_path = os.path.abspath(lib_out)
    lib = ctypes.CDLL(lib_path, winmode=0) if hasattr(os, "add_dll_directory") else ctypes.CDLL(lib_path)

    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_char_p,  # fen
        ctypes.c_int,     # depth / num_simulations
        ctypes.c_char_p,  # out_move buffer
        ctypes.c_int,     # out_len
    ]
    fn.restype = ctypes.c_int
    return fn


def load_standard_alpha_beta():
    """Compile and load the standard C++ alpha-beta engine. Returns the best_move_alpha_beta ctypes fn."""
    src     = os.path.join(STANDARD_DIR, "alpha-beta.cpp")
    lib_out = os.path.join(STANDARD_DIR, "alpha-beta.so")
    return _compile_and_load(src, lib_out, "best_move_alpha_beta")


def load_standard_monte_carlo():
    """Compile and load the standard C++ MCTS engine. Returns the best_move_monte_carlo ctypes fn."""
    src     = os.path.join(STANDARD_DIR, "monte-carlo.cpp")
    lib_out = os.path.join(STANDARD_DIR, "monte-carlo.so")
    return _compile_and_load(src, lib_out, "best_move_monte_carlo")


def load_csl_alpha_beta():
    """Load the CSL alpha-beta engine. Returns the best_move_alpha_beta ctypes fn."""
    pass


def load_csl_monte_carlo():
    """Load the CSL MCTS engine. Returns the best_move_monte_carlo ctypes fn."""
    pass


def load_stockfish_unix(path: str = STOCKFISH_UNIX_BIN):
    """Return a Stockfish instance backed by the Unix binary at `path`."""
    from stockfish import Stockfish
    return Stockfish(path=path)


def load_stockfish_windows(path: str = STOCKFISH_WIN_EXE):
    """Return a Stockfish instance backed by the Windows binary at `path`."""
    from stockfish import Stockfish
    return Stockfish(path=path)


def load_engine(algorithm: str):
    """Return the engine handle for the chosen algorithm."""
    if algorithm == "cpp-alpha-beta":  return load_standard_alpha_beta()
    if algorithm == "cpp-monte-carlo": return load_standard_monte_carlo()
    if algorithm == "csl-alpha-beta":  return load_csl_alpha_beta()
    if algorithm == "csl-monte-carlo": return load_csl_monte_carlo()


def load_stockfish():
    """Load Stockfish using the platform-appropriate binary."""
    if platform.system() == "Windows":
        return load_stockfish_windows()
    return load_stockfish_unix()  # Darwin + Linux
