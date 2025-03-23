# render-build.sh
#!/usr/bin/env bash

set -e

# Install Git LFS
apt-get update && apt-get install -y git-lfs

# Initialize LFS
git lfs install

# Pull LFS files
git lfs pull

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
