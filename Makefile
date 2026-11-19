.PHONY: install test lint check clean docker-build docker-test

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

check: lint test

docker-build:
	docker build -t fin-audit-gym .

docker-test:
	docker run --rm fin-audit-gym pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
