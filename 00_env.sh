#!/usr/bin/env bash
set -eu

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ".venv"
else
    echo "Python virtual environment already exists - skipping creation"
fi

echo "Installing Python dependencies..."
# ----------------------------
# Install Python packages
# ----------------------------
pip show pip > /dev/null 2>&1 || pip install --upgrade pip

packages=(
    numpy
    pandas
    matplotlib
    torch
    h5py
    biopython
    pyfaidx
    pysam
    pybedtools
    logomaker
)

for p in "${packages[@]}"; do
    pip show "$p" > /dev/null 2>&1 || pip install "$p"
done