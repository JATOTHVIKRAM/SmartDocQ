#!/bin/bash
set -e

# Install setuptools and wheel first with specific versions
pip install --upgrade pip
pip install "setuptools>=65.0.0" "wheel>=0.38.0"

# Install packages
pip install -r requirements.txt
