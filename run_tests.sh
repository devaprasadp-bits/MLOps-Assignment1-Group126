#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/pytest tests/ -v --cov=app --cov-report=html --cov-report=term
