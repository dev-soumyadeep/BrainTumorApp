#!/usr/bin/env bash
set -e

# Assume git-lfs is installed via apt.txt
git lfs install

# Pull LFS files
git lfs pull

# Install Python dependencies
pip install -r requirements.txt
