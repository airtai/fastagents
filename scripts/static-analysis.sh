#!/bin/bash
set -e

echo "Running mypy..."
mypy fastagents tests

echo "Running bandit..."
bandit -c pyproject.toml -r fastagents

echo "Running semgrep..."
semgrep scan --config auto --error
