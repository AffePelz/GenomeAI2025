from pyfaidx import Fasta
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os
import h5py

BIN_BED = "data/genome/genome200bp.bed"           # 200bp bins
GENOME_FASTA = "data/genome/Homo_sapiens.chr22.dna.primary_assembly.fa"
OUTPUT_FASTA = "data/genome/sequences_1000bp.fa"
LABEL_FILE = "data/labels/labels_matrix.txt"         # 5-element labels
OUTPUT_H5 = "data/dataset.h5"                        # HDF5 output
# ----------------------------
# Parameters
# ----------------------------
WINDOW_SIZE = 1000
BIN_SIZE = 200
FLANK = (WINDOW_SIZE - BIN_SIZE) // 2  # 400 bp flanks
NUM_LABELS = 5
BATCH_SIZE = 5000  # Adjust depending on memory

def READ_BED(bed_file):
    # Define column names for BED format
    column_names = ['chr', 'start', 'end']
    
    # Read the BED file into a DataFrame
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=column_names, dtype={'chr': str})

    # Make the first two columns of the DataFrame integers
    bed_df['chr'] = bed_df['chr'].str.replace('chr', '', regex=False)
    bed_df['start'] = bed_df['start'].astype(int)
    bed_df['end'] = bed_df['end'].astype(int)

    return bed_df

# ----------------------------
# Load genome
# ----------------------------
print("Loading genome...")
genome = Fasta(GENOME_FASTA)

# ----------------------------
# Load bins
# ----------------------------
print("Loading bins...")
bins = READ_BED(BIN_BED)

# ----------------------------
# Extract 1000bp sequences
# ----------------------------
records = []
for i, row in bins.iterrows():
    chrom = row["chr"]        # Must match FASTA header exactly ("22")
    bin_start = int(row["start"])
    bin_end = int(row["end"])

    win_start = bin_start - FLANK
    win_end = bin_end + FLANK

    seq = ""

    # Pad left if needed
    if win_start < 0:
        seq += "N" * abs(win_start)
        win_start = 0

    # Extract from genome
    seq += str(genome[chrom][win_start:win_end].seq).upper()

    # Pad right if beyond chromosome length
    chrom_len = len(genome[chrom])
    if win_end > chrom_len:
        seq += "N" * (win_end - chrom_len)

    # Ensure exact length
    seq = seq[:WINDOW_SIZE].ljust(WINDOW_SIZE, "N")

    # Save as SeqRecord
    record = SeqRecord(Seq(seq), id=f"{chrom}_{bin_start}_{bin_end}", description="")
    records.append(record)

if __name__ == '__main__':
    # ----------------------------
    # Save sequences to FASTA
    # ----------------------------
    os.makedirs(os.path.dirname(OUTPUT_FASTA), exist_ok=True)
    SeqIO.write(records, OUTPUT_FASTA, "fasta")

    print(f"Saved {len(records)} sequences of length {WINDOW_SIZE} bp to {OUTPUT_FASTA}")