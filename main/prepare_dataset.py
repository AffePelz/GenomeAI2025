from pyfaidx import Fasta
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import matplotlib.pyplot as plt
import os
import h5py

BIN_BED = "data/genome/window_200bp.bed"           # 200bp bins
GENOME_FASTA = "data/genome/hg38.fa"              # Genome FASTA file

OUTPUT_FASTA = "data/genome/sequences_1000bp.fa"
LABEL_FILE = "data/bed_data/labels_matrix.txt"         # 5-element labels
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
# Task 3: Extract 1000bp sequences and save to FASTA
# ----------------------------
records = []
for i, row in bins.iterrows():
    chrom = row["chr"]        # Must match FASTA header exactly ("22")
    bin_start = int(row["start"])
    bin_end = int(row["end"])

    win_start = bin_start - 400
    win_end = bin_end + 400

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

# ----------------------------
# Task 4: Converting DNA to one-hot encoding mapping
# ----------------------------
base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
def one_hot_encode_batch(sequences):
    batch_size = len(sequences)
    arr = np.zeros((batch_size, 1000, 4), dtype=np.uint8)

    for i, seq in enumerate(sequences):
        seq_bytes = np.frombuffer(seq.upper().encode(), dtype='S1')

        arr[i, seq_bytes == b'A', 0] = 1
        arr[i, seq_bytes == b'C', 1] = 1
        arr[i, seq_bytes == b'G', 2] = 1
        arr[i, seq_bytes == b'T', 3] = 1

    return arr

# ----------------------------
# Load labels
# ----------------------------
print("Loading labels...")
y = np.loadtxt(LABEL_FILE, dtype=np.uint8)
num_sequences = len(y)
print(f"Number of sequences: {num_sequences}, each with 5 labels")

if __name__ == '__main__':
    # ----------------------------
    # Task 3: Save sequences to FASTA
    # ----------------------------
    os.makedirs(os.path.dirname(OUTPUT_FASTA), exist_ok=True)
    SeqIO.write(records, OUTPUT_FASTA, "fasta")

    print(f"Saved {len(records)} sequences of length {WINDOW_SIZE} bp to {OUTPUT_FASTA}")

    # ----------------------------
    # Prepare HDF5 file
    # ----------------------------
    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
    with h5py.File(OUTPUT_H5, "w") as h5f:
        X_ds = h5f.create_dataset(
            "X", shape=(num_sequences, WINDOW_SIZE, 4),
            dtype=np.uint8, compression="gzip"
        )
        y_ds = h5f.create_dataset(
            "y", data=y,
            dtype=np.uint8, compression="gzip"
        )

        # ----------------------------
        # Process sequences in batches
        # ----------------------------
        print("Converting sequences to one-hot encoding in batches...")
        seq_batch = []
        idx_start = 0

        for i, record in enumerate(SeqIO.parse(OUTPUT_FASTA, "fasta")):
            seq_batch.append(str(record.seq))

            if len(seq_batch) == BATCH_SIZE or (i + 1) == num_sequences:
                arr = one_hot_encode_batch(seq_batch)
                X_ds[idx_start:idx_start+len(seq_batch)] = arr

                idx_start += len(seq_batch)
                seq_batch = []

                print(f"Processed {idx_start}/{num_sequences} sequences")

        print("One-hot encoding complete.")
        print("Dataset saved to:", OUTPUT_H5)
        print("X shape:", X_ds.shape)
        print("y shape:", y_ds.shape)
    
    # ----------------------------
    # Plot label distribution per track
    # ----------------------------
    print("\nPlotting label distribution...")
    label_sums = np.sum(y, axis=0)  # sequences per track
    tracks = [f"Track {i+1}" for i in range(5)]

    plt.figure(figsize=(8,5))
    plt.bar(tracks, label_sums)
    plt.ylabel("Number of sequences")
    plt.title("Label distribution per track")
    plt.tight_layout()

    plot_path = "label_distribution.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"Label distribution plot saved to {plot_path}")