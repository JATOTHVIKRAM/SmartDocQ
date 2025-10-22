#!/bin/bash
set -e

# Upgrade pip first
pip install --upgrade pip

# Install packages
pip install -r requirements.txt
