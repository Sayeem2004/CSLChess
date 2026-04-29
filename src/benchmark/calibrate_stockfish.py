import argparse
import csv
import os
import platform
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import STOCKFISH_UNIX_BIN


def run_all_positions_under_perf(fens: list, depth: int, sf_path: str, perf_event: str):
    """
    Runs all positions in a single Stockfish process under one perf stat invocation.
    Returns (total_nodes, total_cycles) across all positions.
    Startup overhead is amortized across all searches, keeping cycles/node stable.
    """
    cmd  = ["perf", "stat", "-e", perf_event, sf_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    proc.stdin.write("uci\nisready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if line.strip() == "readyok":
            break

    all_stdout = []
    for fen in fens:
        proc.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
        proc.stdin.flush()
        for line in proc.stdout:
            all_stdout.append(line)
            if line.startswith("bestmove"):
                break

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    _, stderr = proc.communicate()

    total_nodes  = _parse_total_nodes("".join(all_stdout))
    total_cycles = _parse_perf_cycles(stderr, perf_event)
    return total_nodes, total_cycles


def _parse_total_nodes(uci_output: str) -> int:
    """Sum node counts from all 'bestmove' preceding info lines across multiple searches."""
    total  = 0
    last   = 0
    for line in uci_output.splitlines():
        m = re.search(r"\bnodes\s+(\d+)", line)
        if m:
            last = int(m.group(1))
        if line.startswith("bestmove"):
            total += last
            last   = 0
    return total


def _parse_perf_cycles(perf_stderr: str, perf_event: str) -> int | None:
    """
    Extract the counter value from perf stat stderr output.
    perf prints lines like:  1,234,567,890   cycles
    """
    event_base = perf_event.split(":")[0]  # strip modifiers like :u
    for line in perf_stderr.splitlines():
        if event_base in line:
            m = re.search(r"([\d,]+)\s+" + re.escape(event_base), line)
            if m:
                return int(m.group(1).replace(",", ""))
    return None


def calibrate(phase: str, depth: int, sf_path: str, perf_event: str):
    sys_subfolder = "darwin" if platform.system() == "Darwin" else "linux"
    in_path       = os.path.join(DATA_DIR, phase, "puzzles.csv")
    out_dir       = os.path.join(DATA_DIR, phase, sys_subfolder)
    os.makedirs(out_dir, exist_ok=True)
    out_path      = os.path.join(out_dir, "stockfish_calibration.csv")

    if not os.path.exists(in_path):
        print(f"[calibrate] {phase}: puzzles.csv not found")
        return 0, 0

    with open(in_path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f)]

    print(f"[calibrate] {phase}: running {len(fens)} positions at depth {depth} in one process ...")
    total_nodes, total_cycles = run_all_positions_under_perf(fens, depth, sf_path, perf_event)

    if total_nodes == 0:
        print(f"[calibrate] {phase}: parse failed — no nodes counted")
        return 0, 0

    cpn = total_cycles / total_nodes
    print(f"  total nodes={total_nodes:>14,}  total cycles={total_cycles:>16,}  cycles/node={cpn:.2f}")

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["phase", "depth", "total_nodes", "total_cycles", "cycles_per_node"])
        writer.writeheader()
        writer.writerow({"phase": phase, "depth": depth, "total_nodes": total_nodes,
                         "total_cycles": total_cycles, "cycles_per_node": f"{cpn:.2f}"})

    print(f"[calibrate] {phase}: cycles/node = {cpn:.2f}")
    print(f"[calibrate] {phase}: wrote {out_path}")
    return total_nodes, total_cycles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth",      type=int, default=8)
    parser.add_argument("--stockfish",  default=STOCKFISH_UNIX_BIN)
    parser.add_argument("--perf-event", default="cycles",
                        help="perf hardware counter name (CPU-specific)")
    args = parser.parse_args()

    grand_nodes = 0
    grand_cycles = 0
    for phase in PHASES:
        total_nodes, total_cycles = calibrate(phase, args.depth, args.stockfish, args.perf_event)
        grand_nodes += total_nodes
        grand_cycles += total_cycles

    if grand_nodes > 0:
        print(f"\n=== overall cycles/node across all phases: {grand_cycles / grand_nodes:.2f} ===")
