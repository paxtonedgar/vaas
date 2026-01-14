# VaaS Development Makefile

.PHONY: install install-dev test lint format typecheck clean pipeline help

# Default target
help:
	@echo "VaaS Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install runtime dependencies"
	@echo "  make install-dev  Install runtime + tooling dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting (flake8)"
	@echo "  make format       Format code (black)"
	@echo "  make typecheck    Run type checking (mypy)"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline     Run python -m vaas.run_pipeline_v2 with default inputs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	python -m spacy download en_core_web_sm || true

# Testing and Quality
test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# Pipeline Runner
pipeline:
	python -m vaas.run_pipeline_v2 --pdf data/i1099div.pdf --output output --validate

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
