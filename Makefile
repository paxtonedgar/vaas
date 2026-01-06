# VaaS Development Makefile

.PHONY: install install-dev test lint format typecheck clean notebook export-databricks build-wheel help

# Default target
help:
	@echo "VaaS Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package in development mode"
	@echo "  make install-dev  Install with all dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting (flake8)"
	@echo "  make format       Format code (black)"
	@echo "  make typecheck    Run type checking (mypy)"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebook     Start Jupyter Lab"
	@echo ""
	@echo "Databricks:"
	@echo "  make export-databricks  Export notebooks for Databricks"
	@echo "  make build-wheel        Build wheel for Databricks cluster"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,databricks]"
	python -m spacy download en_core_web_sm || true

# Testing and Quality
test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ notebooks/ scripts/

typecheck:
	mypy src/ --ignore-missing-imports

# Notebooks
notebook:
	jupyter lab notebooks/

# Databricks Export
export-databricks:
	python scripts/export_to_databricks.py --all
	@echo ""
	@echo "Notebooks exported to output/databricks/"

build-wheel:
	python scripts/export_to_databricks.py --build-wheel
	@echo ""
	@echo "Wheel built in dist/"

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
