# build_dataset_onehot.py
import numpy as np
from Bio import SeqIO
import h5py
import os

# ----------------------------
# Parameters
# ----------------------------
SEQUENCE_FASTA = "data/genome/sequences_1000bp.fa"  # input 1000bp sequences
LABEL_FILE = "data/labels/labels_matrix.txt"        # 5-element labels
OUTPUT_H5 = "data/dataset.h5"                       # HDF5 output
WINDOW_SIZE = 1000
NUM_LABELS = 5
BATCH_SIZE = 5000  # Adjust depending on memory

# ----------------------------
# One-hot encoding mapping
# ----------------------------
base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode_batch(sequences):
    """
    Vectorized one-hot encoding for a batch of DNA sequences.
    Input: list of sequences (strings)
    Output: numpy array (batch_size, WINDOW_SIZE, 4)
    """
    batch_size = len(sequences)
    arr = np.zeros((batch_size, WINDOW_SIZE, 4), dtype=np.uint8)

    for i, seq in enumerate(sequences):
        # Convert to uppercase bytes for vectorized comparison
        seq_bytes = np.frombuffer(seq.upper().encode(), dtype='S1')

        arr[i, seq_bytes == b'A', 0] = 1
        arr[i, seq_bytes == b'C', 1] = 1
        arr[i, seq_bytes == b'G', 2] = 1
        arr[i, seq_bytes == b'T', 3] = 1
        # N or other letters remain [0,0,0,0]

    return arr

# ----------------------------
# Load labels
# ----------------------------
print("Loading labels...")
y = np.loadtxt(LABEL_FILE, dtype=np.uint8)
num_sequences = len(y)
print(f"Number of sequences: {num_sequences}, each with {NUM_LABELS} labels")

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

    for i, record in enumerate(SeqIO.parse(SEQUENCE_FASTA, "fasta")):
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