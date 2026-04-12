#!/usr/bin/env bash
set -eu

GENOME="data/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
GENOME_CHR22="data/genome/Homo_sapiens.chr22.dna.primary_assembly.fa"
GENOME_FAI="$GENOME.fai"
GENOME_FILE="data/genome/chr22.genome"
GENOME_SPLIT="data/genome/genome200bp.bed"

sudo apt update
sudo apt install -y python3-venv python3-pip samtools bedtools
sudo apt upgrade
sudo apt autoremove

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ".venv"
fi

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

python "main/GPN_tester.py"