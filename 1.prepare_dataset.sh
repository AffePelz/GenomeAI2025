#!/usr/bin/env bash
set -eu

GENOME="data/FASTA/hg38.fa"
GENOME_FAI="$GENOME.fai"

if [ ! -f "$GENOME" ]; then
    echo "Downloading genome..."
    wget -P data/FASTA/ http://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz 
    gunzip data/FASTA/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    mv data/FASTA/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/FASTA/hg38.fa
else
    echo "Genome already downloaded and indexed..."
fi

echo "Activating virtual environment"
. ".venv/bin/activate"

if [ ! -f "data/FASTA/chr22.fa" ]; then
    echo "Extracting chromosome 22 from the FASTA file..."
    samtools faidx data/FASTA/hg38.fa 22 > data/FASTA/chr22.fa
else
    echo "Chromosome 22 already extracted."
fi

python main/GenomeDataset.py