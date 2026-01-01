# Makefile for Heart Disease Prediction MLOps Project

.PHONY: help install setup clean test lint format docker-build docker-run k8s-deploy

help:
	@echo "Available commands:"
	@echo "  make install         - Install all dependencies"
	@echo "  make setup           - Setup project (create directories, prepare data)"
	@echo "  make clean           - Remove generated files and caches"
	@echo "  make test            - Run all tests with coverage"
	@echo "  make lint            - Run linters (flake8, black, isort)"
	@echo "  make format          - Format code with black and isort"
	@echo "  make train           - Train models"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make docker-compose  - Run with docker-compose"
	@echo "  make k8s-deploy      - Deploy to Kubernetes"
	@echo "  make mlflow          - Start MLflow UI"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed models figures logs mlruns
	python scripts/prepare_data.py

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage .pytest_cache
	rm -rf .mypy_cache .tox

test:
	pytest tests/ -v --cov=src --cov=app --cov-report=html --cov-report=term

lint:
	flake8 app/ src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	black --check app/ src/ tests/
	isort --check-only app/ src/ tests/

format:
	black app/ src/ tests/ scripts/
	isort app/ src/ tests/ scripts/

train:
	python scripts/train_model.py --model all --tune

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

docker-compose:
	docker-compose up -d

docker-stop:
	docker-compose down

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml

k8s-delete:
	kubectl delete -f k8s/deployment.yaml

mlflow:
	mlflow ui --backend-store-uri file:./mlruns --port 5000

api:
	uvicorn app.app:app --reload --host 0.0.0.0 --port 8000

test-api:
	python scripts/test_api.py
