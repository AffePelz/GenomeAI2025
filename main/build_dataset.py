import numpy as np
from Bio import SeqIO
import h5py
import os

# ----------------------------
# Parameters
# ----------------------------
SEQUENCE_FASTA = "data/genome/sequences_1000bp.fa"   # 1000bp sequences
LABEL_FILE = "data/labels/labels_matrix.txt"         # 5-element labels
OUTPUT_H5 = "data/dataset.h5"                        # HDF5 output
WINDOW_SIZE = 1000
NUM_LABELS = 5
BATCH_SIZE = 5000  # Adjust depending on memory

# ----------------------------
# One-hot encoding mapping
# ----------------------------
base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode_batch(sequences):
    """Vectorized one-hot encoding for a list of DNA sequences"""
    num_seq = len(sequences)
    arr = np.zeros((num_seq, WINDOW_SIZE, 4), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for j, base in enumerate(seq):
            if base in base_to_idx:
                arr[i, j, base_to_idx[base]] = 1
    return arr

# ----------------------------
# Load labels
# ----------------------------
print("Loading labels...")
y = np.loadtxt(LABEL_FILE, dtype=np.uint8)
num_sequences = len(y)
print(f"Number of labels: {num_sequences}, each with {NUM_LABELS} elements")

# ----------------------------
# Prepare HDF5 file
# ----------------------------
os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
with h5py.File(OUTPUT_H5, "w") as h5f:
    X_ds = h5f.create_dataset("X", shape=(num_sequences, WINDOW_SIZE, 4),
                              dtype=np.uint8, compression="gzip")
    y_ds = h5f.create_dataset("y", data=y, dtype=np.uint8, compression="gzip")

    # ----------------------------
    # Process sequences in batches
    # ----------------------------
    print("Converting sequences to one-hot encoding in batches...")
    seq_batch = []
    idx_start = 0

    for i, record in enumerate(SeqIO.parse(SEQUENCE_FASTA, "fasta")):
        seq_batch.append(str(record.seq))

        # Process batch
        if len(seq_batch) == BATCH_SIZE or (i + 1) == num_sequences:
            arr = one_hot_encode_batch(seq_batch)
            X_ds[idx_start:idx_start+len(seq_batch)] = arr
            idx_start += len(seq_batch)
            seq_batch = []

            print(f"Processed {idx_start}/{num_sequences} sequences")

    # ----------------------------
    # Print dataset shapes inside the context
    # ----------------------------
    print(f"Dataset saved to {OUTPUT_H5}")
    print("X shape:", X_ds.shape)
    print("y shape:", y_ds.shape)

import numpy as np
import matplotlib.pyplot as plt
import h5py

# ----------------------------
# Load dataset
# ----------------------------
H5_FILE = "data/dataset.h5"

with h5py.File(H5_FILE, "r") as f:
    y = f["y"][:]  # shape: (num_sequences, 5)

# ----------------------------
# Compute label distribution
# ----------------------------
label_sums = np.sum(y, axis=0)       # how many sequences have each label
num_sequences = y.shape[0]

for i, count in enumerate(label_sums):
    print(f"Label {i+1}: {count} sequences ({count/num_sequences*100:.2f}%)")

# ----------------------------
# Plot
# ----------------------------
tracks = [f"Track {i+1}" for i in range(y.shape[1])]

plt.figure(figsize=(8,5))
plt.bar(tracks, label_sums, color='skyblue')
plt.ylabel("Number of sequences")
plt.title("Label distribution per track")
plt.savefig('my_plot.png')
plt.show()