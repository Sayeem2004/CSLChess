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

        print_yellow(f"[build] compiling {os.path.basename(src)} with {CXX} ...")
        cmd = [CXX, "-O2", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
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


MPI_DIR = os.path.join(SRC_DIR, "mpi")
MPI_BIN = os.path.join(MPI_DIR, "alpha-beta-mpi")
MPICXX  = os.environ.get("MPICXX", "mpicxx")


def _compile_mpi_binary(src: str, bin_out: str):
    if os.path.exists(bin_out):
        print_yellow(f"[build] {os.path.basename(bin_out)} already exists, skipping compilation\n")
        return

    evaluate_src = os.path.join(STANDARD_DIR, "evaluate.cpp")
    sf_sources   = glob.glob(os.path.join(STOCKFISH_SRC_DIR, "*.cpp")) + \
                   glob.glob(os.path.join(STOCKFISH_SRC_DIR, "syzygy", "*.cpp"))
    sf_sources   = [f for f in sf_sources if os.path.basename(f) != "main.cpp"]

    print_yellow(f"[build] compiling {os.path.basename(src)} with {MPICXX} ...\n")
    cmd = [MPICXX, "-O2", "-DNDEBUG", "-std=c++17", "-fopenmp", f"-I{STANDARD_DIR}",
           src, evaluate_src, *sf_sources, "-o", bin_out]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print_red(f"[build] failed:\n{result.stderr}")
        sys.exit(1)
    print_green(f"[build] succeeded at {os.path.relpath(bin_out)}\n")


class MpiEngine:
    """Persistent MPI engine process. Talks to rank 0 via stdin/stdout.
    The binary must accept lines of the form '<mode> <budget> <fen>' and reply with a UCI move."""

    def __init__(self, src: str, bin_out: str, nranks: int = 2, launcher: str = None):
        _compile_mpi_binary(src, bin_out)
        if launcher is None:
            launcher = "mpirun" if platform.system() == "Darwin" else "srun"
        cmd = [launcher, "-n", str(nranks), bin_out]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=None, text=True)

    def _query(self, mode: str, budget: int, fen: str) -> str:
        self.proc.stdin.write(f"{mode} {budget} {fen}\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def depth(self, fen: str, depth: int)       -> str: return self._query("depth",  depth,      fen)
    def time(self, fen: str, time_ms: int)      -> str: return self._query("time",   time_ms,    fen)
    def cycles(self, fen: str, megacycles: int) -> str: return self._query("cycles", megacycles, fen)

    def close(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        self.proc.communicate()


def load_mpi_alpha_beta(nranks: int = 2, launcher: str = None) -> MpiEngine:
    """Compile (if needed) and launch the MPI alpha-beta engine."""
    src     = os.path.join(MPI_DIR, "alpha-beta-mpi.cpp")
    bin_out = os.path.join(MPI_DIR, "alpha-beta-mpi")
    return MpiEngine(src=src, bin_out=bin_out, nranks=nranks, launcher=launcher)


def load_mpi_monte_carlo(nranks: int = 2, launcher: str = None) -> MpiEngine:
    """Compile (if needed) and launch the MPI MCTS engine."""
    src     = os.path.join(MPI_DIR, "monte-carlo-mpi.cpp")
    bin_out = os.path.join(MPI_DIR, "monte-carlo-mpi")
    return MpiEngine(src=src, bin_out=bin_out, nranks=nranks, launcher=launcher)
