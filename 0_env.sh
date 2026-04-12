#!/usr/bin/env bash
set -eu
sudo apt update
sudo apt install -y python3-venv python3-pip samtools bedtools
sudo apt upgrade
sudo apt autoremove

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ".venv"
else
    echo "Python virtual environment already exists - skipping creation"
fi

echo "Activating virtual entironment..."
. ".venv/bin/activate"

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