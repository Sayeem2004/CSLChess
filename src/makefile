PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)
# export CXX=g++-15 to get commands working on macOS
# salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --qos=interactive --time=01:00:00 --constraint=cpu --account=m4341
# salloc --nodes=2 --ntasks-per-node=1 --cpus-per-task=128 --qos=interactive --time=01:00:00 --constraint=cpu --account=m4341
# source venv/bin/activate

# Installation and setup
.PHONY: venv install clean
venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

clean:
	rm -f standard/alpha-beta.so
	rm -f standard/monte-carlo-rp.so
	rm -f standard/monte-carlo.so
	rm -f standard/perft.so
	rm -f mpi/alpha-beta-mpi
	rm -f mpi/monte-carlo-mpi


# Benchmarking scripts
.PHONY: calibrate-alpha-beta-depth calibrate-alpha-beta-time calibrate-mc-depth calibrate-stockfish
.PHONY: compare-pruning-depths count-nodes count-nodes-ab update-fen
calibrate-alpha-beta-depth: clean
	venv/bin/python -u benchmark/calibrate_alpha_beta_depth.py \
		> ../data/outputs/calibrate/alpha-beta-depth/alpha-beta-depth.txt 2>&1

calibrate-alpha-beta-time: clean
	venv/bin/python -u benchmark/calibrate_alpha_beta_time.py \
		> ../data/outputs/calibrate/alpha-beta-time/alpha-beta-time.txt 2>&1

calibrate-mc-depth: clean
	venv/bin/python -u benchmark/calibrate_monte_carlo_depth.py \
		> ../data/outputs/calibrate/monte-carlo-depth/monte-carlo-depth.txt 2>&1

calibrate-stockfish: clean
	venv/bin/python -u benchmark/calibrate_stockfish.py \
		> ../data/outputs/calibrate/stockfish-cycles/stockfish-cycles.txt 2>&1

compare-pruning-depths: clean
	venv/bin/python -u benchmark/compare_pruning_depths.py \
		> ../data/outputs/benchmark/compare_pruning_depths.txt 2>&1

count-nodes: clean
	venv/bin/python -u benchmark/count_nodes.py \
		> ../data/outputs/benchmark/count_nodes.txt 2>&1

count-nodes-ab: clean
	venv/bin/python -u benchmark/count_nodes_ab.py \
		> ../data/outputs/benchmark/count_nodes_ab.txt 2>&1

update-fen: clean
	venv/bin/python -u benchmark/update_fen.py \
		> ../data/outputs/benchmark/update_fen.txt 2>&1


# Graphing scripts
.PHONY: graph-alpha-beta-depth graph-alpha-beta-time graph-mc-time
.PHONY: graph-search-space
graph-alpha-beta-depth:
	venv/bin/python -u graphs/alpha_beta_depth.py --depth 5

graph-alpha-beta-time:
	venv/bin/python -u graphs/alpha_beta_time.py --budget 500

graph-mc-depth:
	venv/bin/python -u graphs/monte_carlo_depth.py --depth 5

graph-search-space:
	venv/bin/python -u graphs/search_space.py


# Alpha-Beta playing scripts
.PHONY: play-cpp-ab-depth play-cpp-ab-time play-cpp-ab-cycles
.PHONY: play-cpp-ab-depth-elo play-cpp-ab-time-elo play-cpp-ab-cycles-elo
play-cpp-ab-depth:
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --num 100 \
		> ../data/outputs/depth/cpp-alpha-beta-depth.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --verbose \
		> ../data/outputs/depth/cpp-alpha-beta-depth-verbose.txt 2>&1

play-cpp-ab-time:
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --num 100 \
		> ../data/outputs/time/cpp-alpha-beta-time.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --verbose \
		> ../data/outputs/time/cpp-alpha-beta-time-verbose.txt 2>&1

play-cpp-ab-cycles:
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --num 100 \
		> ../data/outputs/cycles/cpp-alpha-beta-cycles.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --verbose \
		> ../data/outputs/cycles/cpp-alpha-beta-cycles-verbose.txt 2>&1

play-cpp-ab-depth-elo: clean
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-depth3-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth3-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth3-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth3-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth3-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth3-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-depth4-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth4-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth4-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth4-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth4-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 4 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth4-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 5 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-depth5-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 5 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth5-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 5 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth5-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 5 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth5-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 5 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth5-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 5 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-depth5-elo.txt 2>&1

play-cpp-ab-time-elo: clean
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time1-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time10-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 100 --elo 1350 --num 25 \
		> ../data/outputs/competitive/cpp-alpha-beta-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 100 --elo 1650 --num 25 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 100 --elo 1950 --num 25 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 100 --elo 2250 --num 25 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 100 --elo 2550 --num 25 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 100 --elo 2850 --num 25 \
		>> ../data/outputs/competitive/cpp-alpha-beta-time100-elo.txt 2>&1

play-cpp-ab-cycles-elo: clean
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles1-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-alpha-beta-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles10-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --elo 1350 --num 50 \
		> ../data/outputs/competitive/cpp-alpha-beta-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --elo 1650 --num 50 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --elo 1950 --num 50 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --elo 2250 --num 50 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --elo 2550 --num 50 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --elo 2850 --num 50 \
		>> ../data/outputs/competitive/cpp-alpha-beta-cycles100-elo.txt 2>&1


# Monte-Carlo playing scripts
.PHONY: play-cpp-mc-depth play-cpp-mc-time play-cpp-mc-cycles
.PHONY: play-cpp-mc-time-elo play-cpp-mc-cycles-elo
play-cpp-mc-depth:
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 4 --num 100 \
		> ../data/outputs/depth/cpp-monte-carlo-depth.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 4 --verbose \
		> ../data/outputs/depth/cpp-monte-carlo-depth-verbose.txt 2>&1

play-cpp-mc-time:
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --num 100 \
		> ../data/outputs/time/cpp-monte-carlo-time.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --verbose \
		> ../data/outputs/time/cpp-monte-carlo-time-verbose.txt 2>&1

play-cpp-mc-cycles:
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --num 100 \
		> ../data/outputs/cycles/cpp-monte-carlo-cycles.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --verbose \
		> ../data/outputs/cycles/cpp-monte-carlo-cycles-verbose.txt 2>&1

play-cpp-mc-time-elo: clean
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 1 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-monte-carlo-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 1 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 1 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 1 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 1 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 1 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time1-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-monte-carlo-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 10 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time10-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 100 --elo 1350 --num 25 \
		> ../data/outputs/competitive/cpp-monte-carlo-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 100 --elo 1650 --num 25 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 100 --elo 1950 --num 25 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 100 --elo 2250 --num 25 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 100 --elo 2550 --num 25 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --time 100 --elo 2850 --num 25 \
		>> ../data/outputs/competitive/cpp-monte-carlo-time100-elo.txt 2>&1

play-cpp-mc-cycles-elo: clean
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 1 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-monte-carlo-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 1 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 1 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 1 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 1 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles1-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 1 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles1-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --elo 1350 --num 100 \
		> ../data/outputs/competitive/cpp-monte-carlo-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --elo 1650 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --elo 1950 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --elo 2250 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --elo 2550 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles10-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 10 --elo 2850 --num 100 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles10-elo.txt 2>&1

	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 100 --elo 1350 --num 50 \
		> ../data/outputs/competitive/cpp-monte-carlo-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 100 --elo 1650 --num 50 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 100 --elo 1950 --num 50 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 100 --elo 2250 --num 50 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 100 --elo 2550 --num 50 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles100-elo.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --cycles 100 --elo 2850 --num 50 \
		>> ../data/outputs/competitive/cpp-monte-carlo-cycles100-elo.txt 2>&1
