PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)
# export CXX=g++-15 to get commands working on macOS

venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

init_outputs:
	mkdir -p outputs/depth
	mkdir -p outputs/time
	mkdir -p outputs/cycles
	mkdir -p outputs/hard


calibrate-stockfish:
	venv/bin/python benchmark/calibrate_stockfish.py

count-nodes:
	venv/bin/python benchmark/count_nodes.py

update-fen:
	venv/bin/python benchmark/update_fen.py


play-cpp-ab-depth: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --num 100 > outputs/depth/cpp-alpha-beta-depth.txt

play-cpp-ab-time: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --num 100 > outputs/time/cpp-alpha-beta-time.txt

play-cpp-ab-cycles: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --cycles 20 --num 100 > outputs/cycles/cpp-alpha-beta-cycles.txt

play-cpp-ab-depth-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --verbose > outputs/depth/cpp-alpha-beta-depth-verbose.txt

play-cpp-ab-time-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --verbose > outputs/time/cpp-alpha-beta-time-verbose.txt

play-cpp-ab-cycles-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-alpha-beta --cycles 20 --verbose > outputs/cycles/cpp-alpha-beta-cycles-verbose.txt

play-cpp-ab-hard: init_outputs
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --skill 0  --num 100 > outputs/hard/cpp-alpha-beta-skill-0.txt
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --skill 10 --num 100 > outputs/hard/cpp-alpha-beta-skill-10.txt
	venv/bin/python -u main.py --algorithm cpp-alpha-beta --depth 3 --skill 20 --num 100 > outputs/hard/cpp-alpha-beta-skill-20.txt


play-cpp-mc-depth: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --num 100 > outputs/depth/cpp-monte-carlo-depth.txt

play-cpp-mc-time: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --time 10 --num 100 > outputs/time/cpp-monte-carlo-time.txt

play-cpp-mc-cycles: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --cycles 20 --num 100 > outputs/cycles/cpp-monte-carlo-cycles.txt

play-cpp-mc-depth-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --verbose > outputs/depth/cpp-monte-carlo-depth-verbose.txt

play-cpp-mc-time-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --time 10 --verbose > outputs/time/cpp-monte-carlo-time-verbose.txt

play-cpp-mc-cycles-verbose: init_outputs
	venv/bin/python main.py --algorithm cpp-monte-carlo --cycles 20 --verbose > outputs/cycles/cpp-monte-carlo-cycles-verbose.txt

play-cpp-mc-hard: init_outputs
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 5 --skill 0  --num 100 > outputs/hard/cpp-monte-carlo-skill-0.txt
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 5 --skill 10 --num 100 > outputs/hard/cpp-monte-carlo-skill-10.txt
	venv/bin/python -u main.py --algorithm cpp-monte-carlo --depth 5 --skill 20 --num 100 > outputs/hard/cpp-monte-carlo-skill-20.txt


.PHONY: venv install
.PHONY: calibrate-stockfish count-nodes update-fen
.PHONY: play-cpp-ab-depth play-cpp-ab-time play-cpp-ab-cycles
.PHONY: play-cpp-ab-depth-verbose play-cpp-ab-time-verbose play-cpp-ab-cycles-verbose
.PHONY: play-cpp-mc-depth play-cpp-mc-time play-cpp-mc-cycles
.PHONY: play-cpp-mc-depth-verbose play-cpp-mc-time-verbose play-cpp-mc-cycles-verbose
