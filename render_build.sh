#!/usr/bin/env bash

set -e

# Git LFS should be installed via apt.txt already
git lfs install

# Pull LFS files
git lfs pull

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
