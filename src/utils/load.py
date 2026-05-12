import subprocess
import platform
import ctypes
import glob
import sys
import os

from utils.random import print_green, print_yellow, print_red


SRC_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDARD_DIR        = os.path.join(SRC_DIR, "standard")
STOCKFISH_DIR       = os.path.join(SRC_DIR, "..", "stockfish")
STOCKFISH_SRC_DIR   = os.path.join(STOCKFISH_DIR, "stockfish-11-src")
STOCKFISH_LINUX_BIN = os.path.join(STOCKFISH_DIR, "stockfish-11-modern")
STOCKFISH_MAC_BIN   = os.path.join(STOCKFISH_DIR, "stockfish-11-modern-mac")
STOCKFISH_WIN_EXE   = os.path.join(STOCKFISH_DIR, "stockfish-11-modern.exe")


MOVE_BUF_LEN = 8 # Longest UCI move is 5 chars (e.g. "e7e8q\0")
CXX          = os.environ.get("CXX", "g++")


def _compile_library(src: str, lib_out: str) -> ctypes.CDLL:
    """Compile src + evaluate.cpp + stockfish code into a shared library and return the CDLL."""
    if os.path.exists(lib_out):
        print_yellow(f"[build] {os.path.basename(lib_out)} already exists, skipping compilation\n")
    else:
        evaluate_src = os.path.join(STANDARD_DIR, "evaluate.cpp")
        sf_sources   = glob.glob(os.path.join(STOCKFISH_SRC_DIR, "*.cpp")) + \
                       glob.glob(os.path.join(STOCKFISH_SRC_DIR, "syzygy", "*.cpp"))
        sf_sources   = [f for f in sf_sources if os.path.basename(f) != "main.cpp"]

        # Attempt to get PAPI flags dynamically, fallback to default
        if platform.system() == "Linux":
            try:
                papi_pkg = ' '.join(subprocess.check_output(["pkg-config", "--cflags", "--libs", "papi"], stderr=subprocess.DEVNULL, text=True).split())
                if papi_pkg:
                    papi_flags = ["-DUSE_PAPI"] + papi_pkg.split()
                else:
                    raise Exception("pkg-config returned empty")
            except Exception:
                # Fallback to absolute paths if available (NERSC setup)
                papi_fallback_dir = "/opt/cray/pe/papi/default"
                if os.path.exists(papi_fallback_dir):
                    papi_flags = [
                        "-DUSE_PAPI",
                        f"-I{papi_fallback_dir}/include",
                        f"-L{papi_fallback_dir}/lib",
                        "-lpapi"
                    ]
                else:
                    papi_flags = ["-DUSE_PAPI", "-lpapi"]
        else:
            papi_flags = []
        print_yellow(f"[build] compiling {os.path.basename(src)} with {CXX} ...")
        cmd = [CXX, "-O2", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
               *papi_flags, src, evaluate_src, *sf_sources, "-o", lib_out]

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
        ctypes.c_int,     # budget (depth / time_ms / megacycle_budget)
        ctypes.c_char_p,  # out_move buffer
        ctypes.c_int,     # out_len
    ]
    fn.restype = ctypes.c_int
    return fn


def load_standard_alpha_beta() -> dict:
    """Compile and load the standard C++ alpha-beta engine.
    Returns a dict with keys 'depth', 'time', 'cycles' mapping to their ctypes fns."""
    src     = os.path.join(STANDARD_DIR, "alpha-beta.cpp")
    lib_out = os.path.join(STANDARD_DIR, "alpha-beta.so")
    lib = _compile_library(src, lib_out)
    return {
        "depth":  _bind_fn(lib, "best_move_alpha_beta_depth"),
        "time":   _bind_fn(lib, "best_move_alpha_beta_time"),
        "cycles": _bind_fn(lib, "best_move_alpha_beta_cycles"),
    }


def load_standard_monte_carlo() -> dict:
    """Compile and load the standard C++ MCTS engine.
    Returns a dict with keys 'depth', 'time', 'cycles' mapping to their ctypes fns."""
    src     = os.path.join(STANDARD_DIR, "monte-carlo.cpp")
    lib_out = os.path.join(STANDARD_DIR, "monte-carlo.so")
    lib = _compile_library(src, lib_out)
    return {
        "depth":  _bind_fn(lib, "best_move_monte_carlo_depth"),
        "time":   _bind_fn(lib, "best_move_monte_carlo_time"),
        "cycles": _bind_fn(lib, "best_move_monte_carlo_cycles"),
    }


def load_standard_monte_carlo_rp() -> dict:
    """Compile and load the standard C++ MCTS engine (root parallel).
    Returns a dict with keys 'depth', 'time', 'cycles' mapping to their ctypes fns."""
    src     = os.path.join(STANDARD_DIR, "monte-carlo-rp.cpp")
    lib_out = os.path.join(STANDARD_DIR, "monte-carlo-rp.so")
    lib = _compile_library(src, lib_out)
    return {
        "depth":  _bind_fn(lib, "best_move_monte_carlo_depth"),
        "time":   _bind_fn(lib, "best_move_monte_carlo_time"),
        "cycles": _bind_fn(lib, "best_move_monte_carlo_cycles"),
    }


def load_perft():
    """Compile and load perft + alpha-beta node counter.
    Returns a dict with:
      'perft':      count_nodes_depth(fen, depth) -> long long
      'alpha_beta': count_nodes_alpha_beta_depth(fen, depth) -> long long
    """
    src     = os.path.join(STANDARD_DIR, "perft.cpp")
    lib_out = os.path.join(STANDARD_DIR, "perft.so")
    lib = _compile_library(src, lib_out)

    perft_fn          = lib.count_nodes_depth
    perft_fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
    perft_fn.restype  = ctypes.c_longlong

    ab_fn          = lib.count_nodes_alpha_beta_depth
    ab_fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
    ab_fn.restype  = ctypes.c_longlong

    return {"perft": perft_fn, "alpha_beta": ab_fn}


# Stockfish 11 UCI_Elo range: 1350–2850 (from `uci` command output).
# Skill Level (0-20) has no official ELO mapping; use --elo for precise ELO targeting.
SF_ELO_MIN = 1350
SF_ELO_MAX = 2850


def _load_stockfish(elo: int, path: str):
    from stockfish import Stockfish
    elo = max(SF_ELO_MIN, min(SF_ELO_MAX, elo))
    print_yellow(f"[stockfish] loaded at {elo} ELO (UCI_LimitStrength)\n")
    return Stockfish(path=path, parameters={"UCI_LimitStrength": True, "UCI_Elo": elo})


def load_stockfish_linux(elo: int = SF_ELO_MIN, path: str = STOCKFISH_LINUX_BIN):
    """Return a Stockfish instance backed by the Linux binary."""
    return _load_stockfish(elo, path)


def load_stockfish_mac(elo: int = SF_ELO_MIN, path: str = STOCKFISH_MAC_BIN):
    """Return a Stockfish instance backed by the macOS binary."""
    return _load_stockfish(elo, path)


def load_stockfish_windows(elo: int = SF_ELO_MIN, path: str = STOCKFISH_WIN_EXE):
    """Return a Stockfish instance backed by the Windows binary."""
    return _load_stockfish(elo, path)


def load_engine(algorithm: str) -> dict:
    """Return a dict of {budget_type: ctypes_fn} for the chosen algorithm."""
    if algorithm == "cpp-alpha-beta":     return load_standard_alpha_beta()
    if algorithm == "cpp-monte-carlo":    return load_standard_monte_carlo()
    if algorithm == "cpp-monte-carlo-rp": return load_standard_monte_carlo_rp()


def load_stockfish(elo: int = SF_ELO_MIN):
    """Load Stockfish using the platform-appropriate binary."""
    if platform.system() == "Windows": return load_stockfish_windows(elo)
    if platform.system() == "Darwin":  return load_stockfish_mac(elo)
    return load_stockfish_linux(elo)
