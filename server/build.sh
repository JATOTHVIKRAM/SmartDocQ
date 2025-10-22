#!/bin/bash
set -e

# Install setuptools and wheel first
pip install --upgrade pip setuptools wheel

# Install packages
pip install -r requirements.txt
