# Makefile for ANTsXMM
# Modern BIDS orchestration and multimodal image processing wrapper

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= ruff

.PHONY: help install install-dev develop install-user test test-modalities test-strict test-cov lint lint-fix format format-fix compile audit check build clean clean-all cli-check

# Default target
help:
	@echo "ANTsXMM Development & Maintenance Tasks"
	@echo "========================================="
	@echo "  make install           Install antsxmm into current Python environment"
	@echo "  make install-dev       Install antsxmm in editable mode with test dependencies (-e \".[test]\")"
	@echo "  make install-user      Install antsxmm into user site-packages (--user .)"
	@echo "  make test              Run full test suite via pytest"
	@echo "  make test-modalities   Run modalities unit test suite"
	@echo "  make test-strict       Run pytest treating deprecation warnings as fatal errors"
	@echo "  make test-cov          Run pytest with coverage report"
	@echo "  make lint              Run code quality and static checks using ruff"
	@echo "  make lint-fix          Run ruff and automatically apply safe fixes"
	@echo "  make format            Check code formatting with ruff"
	@echo "  make format-fix        Format code with ruff"
	@echo "  make compile           Verify bytecode compilation without SyntaxWarnings"
	@echo "  make audit             Run full validation gate (compile + lint + test)"
	@echo "  make check             Alias for make audit"
	@echo "  make build             Build source distribution and wheel"
	@echo "  make clean             Remove build artifacts, caches, and compiled bytecode"
	@echo "  make clean-all         Remove all caches, build artifacts, and output directories"
	@echo "  make cli-check         Verify that the antsxmm CLI executable is responsive"

install:
	$(PIP) install .

install-dev: develop

develop:
	$(PIP) install -e ".[test]"

install-user:
	$(PIP) install --user .

test:
	$(PYTEST) -v tests/

test-modalities:
	$(PYTEST) -v tests/test_modalities.py

test-strict:
	$(PYTEST) -v -W error::DeprecationWarning -W error::FutureWarning tests/

test-cov:
	$(PYTEST) -v --cov=antsxmm --cov-report=term-missing tests/

lint:
	$(RUFF) check antsxmm tests

lint-fix:
	$(RUFF) check --fix antsxmm tests

format:
	$(RUFF) format --check antsxmm tests

format-fix:
	$(RUFF) format antsxmm tests

compile:
	$(PYTHON) -W error::SyntaxWarning -m compileall antsxmm tests

audit: compile lint test

check: audit

build: clean
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*$$py.class" -delete

clean-all: clean
	rm -rf out/

cli-check:
	$(PYTHON) -m antsxmm.cli --help
