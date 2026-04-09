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
)

for p in "${packages[@]}"; do
    pip show "$p" > /dev/null 2>&1 || pip install "$p"
done
# ----------------------------
# Task 1: Extract chromosome 22 and prepare genome file for bedtools
# ----------------------------
if [ ! -f "$GENOME_FAI" ]; then
    echo "Running samtools faidx..."
    samtools faidx "$GENOME" 22 > "$GENOME_CHR22"
else
    echo "Chromosome 22 sequence already exists - skipping samtools faidx"
fi

awk '$1 == "22"' "$GENOME_FAI" | cut -f1,2 > "$GENOME_FILE"

if [ ! -f "$GENOME_SPLIT" ]; then
    echo "Splitting chromosome 22 into 200bp bins..."
    bedtools makewindows -g "$GENOME_FILE" -w 200 > "$GENOME_SPLIT"
else
    echo "Genome already split into bins - skipping bedtools makewindows"
fi
# ----------------------------
# Task 2: Generate 5-element label vector
# ----------------------------
echo "Labeling bins with BED files..."
mkdir -p data/labels

cmds=()

for f in data/bed_data/*.bed; do
  cmds+=("<(bedtools intersect -a \"$GENOME_SPLIT\" -b \"$f\" -f 0.5 -c | awk '{print (\$NF>0)}')")
done

# Use eval to expand process substitutions
eval paste "${cmds[@]}" > data/labels/labels_matrix.txt

echo "Labeling complete."

python "main/prepare_dataset.py"
