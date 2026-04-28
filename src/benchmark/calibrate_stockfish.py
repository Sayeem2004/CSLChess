import argparse
import csv
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import STOCKFISH_UNIX_BIN


def run_stockfish_under_perf(fen: str, depth: int, sf_path: str, perf_event: str):
    """
    Spawns `perf stat -e <event> stockfish`, sends a fixed-depth search,
    and returns (nodes, flops). Returns (None, None) on parse failure.
    """
    cmd  = ["perf", "stat", "-e", perf_event, sf_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    proc.stdin.write("uci\nisready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if line.strip() == "readyok":
            break

    proc.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
    proc.stdin.flush()

    stdout_lines = []
    for line in proc.stdout:
        stdout_lines.append(line)
        if line.startswith("bestmove"):
            break

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    _, stderr = proc.communicate()

    nodes = _parse_nodes("".join(stdout_lines))
    flops = _parse_perf_flops(stderr, perf_event)
    return nodes, flops


def _parse_nodes(uci_output: str) -> int | None:
    """Extract node count from the last 'info depth ... nodes N' line."""
    nodes = None
    for line in uci_output.splitlines():
        m = re.search(r"\bnodes\s+(\d+)", line)
        if m:
            nodes = int(m.group(1))
    return nodes


def _parse_perf_flops(perf_stderr: str, perf_event: str) -> int | None:
    """
    Extract the counter value from perf stat stderr output.
    perf prints lines like:  1,234,567,890   fp_ret_sse_avx_ops.all
    """
    event_base = perf_event.split(":")[0]  # strip modifiers like :u
    for line in perf_stderr.splitlines():
        if event_base in line:
            m = re.search(r"([\d,]+)\s+" + re.escape(event_base), line)
            if m:
                return int(m.group(1).replace(",", ""))
    return None


def calibrate(phase: str, depth: int, sf_path: str, perf_event: str):
    in_path  = os.path.join(DATA_DIR, phase, "updatedFEN.csv")
    out_path = os.path.join(DATA_DIR, phase, "stockfish_flops_per_node.csv")

    if not os.path.exists(in_path):
        print(f"[calibrate] {phase}: updatedFEN.csv not found, run update_fen.py first")
        return 0, 0

    with open(in_path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f)]

    print(f"[calibrate] {phase}: running {len(fens)} positions at depth {depth} ...")

    rows            = []
    total_nodes     = 0
    total_flops     = 0
    failed          = 0

    for i, fen in enumerate(fens):
        nodes, flops = run_stockfish_under_perf(fen, depth, sf_path, perf_event)
        if nodes is None or flops is None or nodes == 0:
            print(f"  pos {i}: parse failed (nodes={nodes}, flops={flops})")
            failed += 1
            continue
        fpn = flops / nodes
        rows.append({"FEN": fen, "nodes": nodes, "flops": flops, "flops_per_node": f"{fpn:.2f}"})
        total_nodes += nodes
        total_flops += flops
        print(f"  pos {i}: nodes={nodes:>12,}  flops={flops:>14,}  flops/node={fpn:.2f}")

    if rows:
        overall_fpn = total_flops / total_nodes
        rows.append({"FEN": "AVERAGE", "nodes": total_nodes // len(rows),
                     "flops": total_flops // len(rows),
                     "flops_per_node": f"{overall_fpn:.2f}"})

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["FEN", "nodes", "flops", "flops_per_node"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        print(f"[calibrate] {phase}: overall flops/node = {overall_fpn:.2f}  "
              f"({failed} positions failed)")
    print(f"[calibrate] {phase}: wrote {out_path}")
    return total_nodes, total_flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth",      type=int, default=10)
    parser.add_argument("--stockfish",  default=STOCKFISH_UNIX_BIN)
    parser.add_argument("--perf-event", default="fp_ret_sse_avx_ops.all",
                        help="perf hardware counter name (CPU-specific)")
    args = parser.parse_args()

    grand_nodes = 0
    grand_flops = 0
    for phase in PHASES:
        total_nodes, total_flops = calibrate(phase, args.depth, args.stockfish, args.perf_event)
        grand_nodes += total_nodes
        grand_flops += total_flops

    if grand_nodes > 0:
        print(f"\n=== overall flops/node across all phases: {grand_flops / grand_nodes:.2f} ===")
