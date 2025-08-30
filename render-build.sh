#!/usr/bin/env bash
set -o errexit  # stop on error

# Make sure build tools are ready before any package builds
pip install --upgrade pip setuptools wheel

# Now install your requirements
pip install -r requirements.txt
