#!/usr/bin/env bash
set -e

# Initialize Git LFS (git-lfs is installed via apt.txt or Dockerfile)
git lfs install

# Pull Git LFS files
git lfs pull

# Install Python dependencies
pip install -r requirements.txt