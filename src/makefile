PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)
# export CXX=g++-15 to get commands working on macOS
# salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --qos=interactive --time=01:00:00 --constraint=cpu --account=m4341
# source venv/bin/activate

venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

init_outputs:
	mkdir -p outputs/competitive
	mkdir -p outputs/cycles
	mkdir -p outputs/depth
	mkdir -p outputs/hard
	mkdir -p outputs/time

clean:
	rm -f standard/alpha-beta.so
	rm -f standard/monte-carlo.so


calibrate-alpha-beta-branch: clean
	venv/bin/python benchmark/calibrate_alpha_beta_branch.py \
	> outputs/calibrate/alpha-beta-branch/alpha-beta-branch.txt 2>&1

calibrate-alpha-beta-depth: clean
	venv/bin/python benchmark/calibrate_alpha_beta_depth.py \
	> outputs/calibrate/alpha-beta-depth/alpha-beta-depth.txt 2>&1

calibrate-stockfish:
	venv/bin/python benchmark/calibrate_stockfish.py

compare-depths:
	venv/bin/python benchmark/compare_depths.py

count-nodes:
	venv/bin/python benchmark/count_nodes.py

count-nodes-ab:
	venv/bin/python benchmark/count_nodes_ab.py

update-fen:
	venv/bin/python benchmark/update_fen.py


graph-alpha-beta-branch:
	venv/bin/python graphs/alpha_beta_branch.py --budget 500

graph-alpha-beta-depth:
	venv/bin/python graphs/alpha_beta_depth.py --depth 5

graph-search-space:
	venv/bin/python graphs/search_space.py


play-cpp-ab-depth: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --num 100 > outputs/depth/cpp-alpha-beta-depth.txt 2>&1

play-cpp-ab-time: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --num 100 > outputs/time/cpp-alpha-beta-time.txt 2>&1

play-cpp-ab-cycles: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --cycles 20 --num 100 > outputs/cycles/cpp-alpha-beta-cycles.txt 2>&1

play-cpp-ab-depth-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --verbose > outputs/depth/cpp-alpha-beta-depth-verbose.txt 2>&1

play-cpp-ab-time-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --verbose > outputs/time/cpp-alpha-beta-time-verbose.txt 2>&1

play-cpp-ab-cycles-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --cycles 20 --verbose > outputs/cycles/cpp-alpha-beta-cycles-verbose.txt 2>&1

play-cpp-ab-hard: init_outputs
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --skill 0  --num 100 > outputs/hard/cpp-alpha-beta-skill-0.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --skill 10 --num 100 > outputs/hard/cpp-alpha-beta-skill-10.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --skill 20 --num 100 > outputs/hard/cpp-alpha-beta-skill-20.txt 2>&1

play-cpp-ab-competitive: init_outputs
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 1 --num 100 > outputs/competitive/cpp-alpha-beta-1ms.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --time 10 --num 100 > outputs/competitive/cpp-alpha-beta-10ms.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 1 --num 100 > outputs/competitive/cpp-alpha-beta-1MC.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 10 --num 100 > outputs/competitive/cpp-alpha-beta-10MC.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --cycles 100 --num 100 > outputs/competitive/cpp-alpha-beta-100MC.txt 2>&1

play-cpp-mc-depth: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --num 100 > outputs/depth/cpp-monte-carlo-depth.txt 2>&1

play-cpp-mc-time: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --time 10 --num 100 > outputs/time/cpp-monte-carlo-time.txt 2>&1

play-cpp-mc-cycles: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --cycles 20 --num 100 > outputs/cycles/cpp-monte-carlo-cycles.txt 2>&1

play-cpp-mc-depth-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --verbose > outputs/depth/cpp-monte-carlo-depth-verbose.txt 2>&1

play-cpp-mc-time-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --time 10 --verbose > outputs/time/cpp-monte-carlo-time-verbose.txt 2>&1

play-cpp-mc-cycles-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --cycles 20 --verbose > outputs/cycles/cpp-monte-carlo-cycles-verbose.txt 2>&1

play-cpp-mc-hard: init_outputs
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 5 --skill 0  --num 100 > outputs/hard/cpp-monte-carlo-skill-0.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 5 --skill 10 --num 100 > outputs/hard/cpp-monte-carlo-skill-10.txt 2>&1
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 5 --skill 20 --num 100 > outputs/hard/cpp-monte-carlo-skill-20.txt 2>&1


.PHONY: venv install
.PHONY: calibrate-stockfish count-nodes update-fen
.PHONY: play-cpp-ab-depth play-cpp-ab-time play-cpp-ab-cycles
.PHONY: play-cpp-ab-depth-verbose play-cpp-ab-time-verbose play-cpp-ab-cycles-verbose
.PHONY: play-cpp-mc-depth play-cpp-mc-time play-cpp-mc-cycles
.PHONY: play-cpp-mc-depth-verbose play-cpp-mc-time-verbose play-cpp-mc-cycles-verbose
