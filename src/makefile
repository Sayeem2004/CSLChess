PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)

venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install -r requirements.txt


calibrate-stockfish:
	venv/bin/python benchmark/calibrate_stockfish.py

count-nodes:
	venv/bin/python benchmark/count_nodes.py

update-fen:
	venv/bin/python benchmark/update_fen.py


play-cpp-ab-depth:
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --num 100

play-cpp-ab-time:
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --num 100

play-cpp-ab-flops:
	venv/bin/python main.py --algorithm cpp-alpha-beta --flops 20 --num 100

play-cpp-ab-depth-verbose:
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --verbose

play-cpp-ab-time-verbose:
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --verbose

play-cpp-ab-flops-verbose:
	venv/bin/python main.py --algorithm cpp-alpha-beta --flops 20 --verbose


play-cpp-mc-depth:
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --num 100

play-cpp-mc-time:
	venv/bin/python main.py --algorithm cpp-monte-carlo --time 10 --num 100

play-cpp-mc-flops:
	venv/bin/python main.py --algorithm cpp-monte-carlo --flops 20 --num 100

play-cpp-mc-depth-verbose:
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --verbose

play-cpp-mc-time-verbose:
	venv/bin/python main.py --algorithm cpp-monte-carlo --time 10 --verbose

play-cpp-mc-flops-verbose:
	venv/bin/python main.py --algorithm cpp-monte-carlo --flops 20 --verbose


.PHONY: venv install
.PHONY: calibrate-stockfish count-nodes update-fen
.PHONY: play-cpp-ab-depth play-cpp-ab-time play-cpp-ab-flops
.PHONY: play-cpp-ab-depth-verbose play-cpp-ab-time-verbose play-cpp-ab-flops-verbose
.PHONY: play-cpp-mc-depth play-cpp-mc-time play-cpp-mc-flops
.PHONY: play-cpp-mc-depth-verbose play-cpp-mc-time-verbose play-cpp-mc-flops-verbose
