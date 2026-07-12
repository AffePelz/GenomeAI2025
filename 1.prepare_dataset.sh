#!/usr/bin/env bash
set -eu

GENOME="data/genome/hg38.fa"
GENOME_FAI="$GENOME.fai"
GENOME_SPLIT="data/genome/window_200bp.bed"

if [ ! -f "$GENOME" ]; then
    echo "Downloading genome..."
    wget -P data/genome/ http://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz 
    gunzip data/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    mv data/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/genome/hg38.fa
else
    echo "Genome already downloaded and indexed..."
fi

echo "Activating virtual environment"
. ".venv/bin/activate"

echo "Extracting the regions of the FASTA file..."
samtools faidx data/genome/hg38.fa 22 > data/genome/chr22.fa

# ---------------------
# Task 1: Generating 200bp coordinate windows for chromosome 22
# ---------------------
if [ ! -f "$GENOME_SPLIT" ]; then
    echo "Generating 200bp coordinate windows for chromosome 22..."
    bedtools makewindows -g <(awk '$1 == "22"' "$GENOME_FAI" | cut -f1,2) -w 200 > "$GENOME_SPLIT"
    bedtools getfasta -fi data/genome/chr22.fa -bed "$GENOME_SPLIT" > data/genome/chr22_200bp.fa
else
    echo "Genome bins already created."
fi

# ---------------------
# Task 2: Labeling all five BED files
# ---------------------
echo "Labeling BED files..."
mkdir -p data/labels

for f in data/bed_data/*.bed; do
    out="data/labels/$(basename "$f").txt"
    
    bedtools intersect -a "$GENOME_SPLIT" -b "$f" -f 0.5 -c \
    | awk '{print ($NF>0)}' > "$out"
done

paste data/labels/*.txt > data/bed_data/labels_matrix.txt
rm -rf data/labels


python main/prepare_dataset.py