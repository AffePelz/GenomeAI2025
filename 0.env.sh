#!/usr/bin/env bash
set -eu

echo "================================================================="
echo " System Updates & Dependencies"
echo "================================================================="
sudo apt update
# Added build-essential, g++, and python3-dev
sudo apt install -y build-essential g++ python3-dev python3-venv python3-pip samtools bedtools zlib1g-dev
sudo apt upgrade -y
sudo apt autoremove -y

echo "================================================================="
echo " Python Virtual Environment Setup"
echo "================================================================="
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ".venv"
else
    echo "Python virtual environment already exists - skipping creation"
fi

echo "Activating virtual environment..."
. ".venv/bin/activate"

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "================================================================="
echo " Installing Python Dependencies"
echo "================================================================="

# Core packages required for data processing and training
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
    datasets
    zstandard
    tables
    wandb
    python-dotenv
    scikit-learn
    fire
    accelerate
    numdifftools
)

# Install missing packages selectively
for p in "${packages[@]}"; do
    if pip show "$p" > /dev/null 2>&1; then
        echo "  [✓] $p is already installed"
    else
        echo "  [+] Installing $p..."
        pip install "$p"
    fi
done