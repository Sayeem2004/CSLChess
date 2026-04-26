PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)

venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/pip install -r requirements.txt

play-cpp-ab-depth:
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --num 100

play-cpp-ab-time:
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --num 100

play-cpp-ab-depth-verbose:
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 3 --verbose

play-cpp-ab-time-verbose:
	venv/bin/python main.py --algorithm cpp-alpha-beta --time 10 --verbose

play-cpp-mc-depth:
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --num 100

play-cpp-mc-verbose:
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --verbose

.PHONY: venv install
