# Makefile for hf-platform

PYTHON := python3.11
DOCKER_IMG := hf-platform/worker
TAG := $(shell git rev-parse --short HEAD)

.PHONY: help fmt lint test cov docker-run docker-build k8s-apply k8s-delete clean

help:
	@echo "Available commands:"
	@echo "  make fmt           - Auto-format code with black"
	@echo "  make lint          - Run Ruff linting"
	@echo "  make test          - Run tests with pytest"
	@echo "  make cov           - Run tests with coverage"
	@echo "  make docker-build  - Build docker image"
	@echo "  make docker-run    - Run docker container locally"
	@echo "  make k8s-apply     - Apply all k8s manifests"
	@echo "  make k8s-delete    - Delete all k8s resources"
	@echo "  make clean         - Remove caches, pyc, build artifacts"

fmt:
	$(PYTHON) -m black .

lint:
	ruff check .

test:
	pytest -q

cov:
	pytest --cov=platform --cov=workers --cov=research --maxfail=2 -q

docker-build:
	docker build -f docker/worker.Dockerfile -t $(DOCKER_IMG):$(TAG) .

docker-run:
	docker run --rm -it -e REDIS_HOST=localhost -p 8000:8000 $(DOCKER_IMG):$(TAG)

k8s-apply:
	kubectl apply -k k8s/

k8s-delete:
	kubectl delete -k k8s/ || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache htmlcov dist build