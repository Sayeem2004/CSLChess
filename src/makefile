PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)

venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/pip install -r requirements.txt

play-cpp-alpha-beta:
	venv/bin/python main.py --algorithm cpp-alpha-beta --depth 5 --num 100

play-cpp-monte-carlo:
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --num 100

play-cpp-monte-carlo-verbose:
	venv/bin/python main.py --algorithm cpp-monte-carlo --depth 5 --verbose

.PHONY: venv install
