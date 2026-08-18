# Makefile

SHELL := /bin/bash

PIP ?= /path/envs/DPI/bin/pip
PYTHON ?= /path/envs/DPI/bin/python
RUNS ?= runs

install:
	@echo "[install] start"
	$(PIP) install -e . --upgrade
	@rm -rf ./src/DPI.egg-info
	@echo "[install] done"

clean:
	@echo "[clean] start"
	@test -n "$(RUNS)" && rm -rf "$(RUNS)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "[clean] done"

uninstall:
	@echo "[uninstall] start"
	$(PIP) uninstall -y DPI
	@echo "[uninstall] done"
