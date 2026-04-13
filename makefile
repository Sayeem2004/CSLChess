PYTHON := $(shell command -v python3.10 || command -v python3 || command -v python)

venv:
	$(PYTHON) -m venv venv

install: venv
	venv/bin/pip install -r requirements.txt

.PHONY: venv install
