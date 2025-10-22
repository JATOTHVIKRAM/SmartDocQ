#!/bin/bash
set -e

# Upgrade pip first
pip install --upgrade pip

# Install packages with specific flags to avoid compilation issues
pip install --no-cache-dir --only-binary=all -r requirements.txt

# If the above fails, try without the only-binary flag
if [ $? -ne 0 ]; then
    echo "Retrying with compilation allowed..."
    pip install --no-cache-dir -r requirements.txt
fi
