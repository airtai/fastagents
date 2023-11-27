#!/bin/bash

echo "Running pyup_dirs..."
pyup_dirs --py38-plus --recursive fastagents tests

echo "Running ruff..."
ruff fastagents tests --fix

echo "Running black..."
black fastagents tests
