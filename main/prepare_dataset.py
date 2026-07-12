import os
import numpy as np
import h5py
import pandas as pd
from Bio import SeqIO
from pybedtools import BedTool
import matplotlib.pyplot as plt
# ----------------------------
# Files and parameters
# ----------------------------
FASTA_FILE = "data/genome/chr22.fa"
BED_FILES = ["data/bed_data/ENCFF052RRA.bed",
             "data/bed_data/ENCFF053BLB.bed",
             "data/bed_data/ENCFF057CRD.bed",
             "data/bed_data/ENCFF060JHQ.bed",
             "data/bed_data/ENCFF078AMJ.bed"]
OUTPUT_H5 = "data/dataset.h5"

CHROM = "chr22"
BIN_SIZE = 200
FLANK_SIZE = 400
WINDOW_SIZE = 1000

def READ_BED(bed_file):
    column_names = ['chr', 'start', 'end']
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=column_names, usecols=[0, 1, 2], dtype={'chr': str})

    bed_df['chr'] = bed_df['chr'].str.replace('chr', '', regex=False)
    bed_df['start'] = bed_df['start'].astype(int)
    bed_df['end'] = bed_df['end'].astype(int)

    return bed_df

def load_chromosome_sequence(fasta_path):
    print(f"Loading sequence from {fasta_path}...")
    record = SeqIO.read(fasta_path, "fasta") # We have only one record, chromosome 22
    print(f"Successfully loaded sequence ID: {record.id}")

    return str(record.seq)

# ----------------------------
# Task 4: Converting DNA to one-hot encoding mapping
# ----------------------------
def one_hot_encode(sequences):
    batch_size = len(sequences)
    seq_len = len(sequences[0])  # 1000 bp

    arr = np.zeros((batch_size, seq_len, 4), dtype=np.uint8)

    for i, seq in enumerate(sequences):
        seq = seq.upper()
        seq_bytes = np.frombuffer(seq.encode(), dtype='S1')

        # Map A->0, C->1, G->2, T->3
        arr[i, seq_bytes == b'A', 0] = 1
        arr[i, seq_bytes == b'C', 1] = 1
        arr[i, seq_bytes == b'G', 2] = 1
        arr[i, seq_bytes == b'T', 3] = 1

    return arr

def build_pipeline():
    # 1. Load sequence up front
    chrom_seq = load_chromosome_sequence(FASTA_FILE)
    chrom_size = len(chrom_seq)
    print(f"{CHROM} size: {chrom_size} bp")
    print("Generating 200bp genome bins...")

    bins = []
    for start in range(0, chrom_size, BIN_SIZE):
        end = min(start + BIN_SIZE, chrom_size)
        bins.append((CHROM, start, end))
    
    bins_bed = BedTool(bins)
    num_bins = len(bins_bed)
    print(f"Total bins generated: {num_bins}")

    # 2. Label Generation (Matrix shape: [num_bins, 5])
    labels_matrix = np.zeros((num_bins, len(BED_FILES)), dtype=np.int8)

    for idx, bed_path in enumerate(BED_FILES):
        print(f"Processing overlaps for {bed_path}...")
        if not os.path.exists(bed_path):
            raise FileNotFoundError(f"Could not find BED file: {bed_path}")
            
        bed_track = BedTool(bed_path)
        intersect = bins_bed.intersect(bed_track, wo=True, nonamecheck=True)
        
        for feature in intersect:
            bin_start = int(feature[1])
            overlap_bp = int(feature[-1])
            
            if overlap_bp >= 100:
                bin_idx = bin_start // BIN_SIZE
                if bin_idx < num_bins:
                    labels_matrix[bin_idx, idx] = 1
    
    # 3. Input Sequence Extraction and One-Hot Encoding
    print(f"Writing dataset to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as hf:
        x_ds = hf.create_dataset('X', (num_bins, WINDOW_SIZE, 4), dtype=np.uint8, chunks=(128, WINDOW_SIZE, 4), compression="gzip")
        y_ds = hf.create_dataset('y', data=labels_matrix, chunks=(128, 5), compression="gzip")

        batch_sequences = []
        batch_indices = []
        batch_size_limit = 10000

        for i, b in enumerate(bins):
            _, start, end = b
            
            # Center the 1000bp window over the 200bp bin
            window_start = start - FLANK_SIZE
            window_end = end + FLANK_SIZE
            
            # 1. Track how much we drop off the left edge
            left_pad = 0
            if window_start < 0:
                left_pad = abs(window_start)
                window_start = 0
                
            # 2. Slice whatever is available within safe bounds
            if window_end > chrom_size:
                window_end = chrom_size
            
            seq_chunk = chrom_seq[window_start:window_end]
            
            # 3. Form the initial sequence with left padding
            final_seq = ('N' * left_pad) + seq_chunk
            
            # 4. Dynamically pad the right side to guarantee exactly 1000bp
            # This cleanly handles both edge drop-off AND truncated end-of-chromosome bins!
            if len(final_seq) < WINDOW_SIZE:
                right_pad = WINDOW_SIZE - len(final_seq)
                final_seq = final_seq + ('N' * right_pad)
            
            batch_sequences.append(final_seq)
            batch_indices.append(i)

            # Write out batch when limit reached or at the final bin
            if len(batch_sequences) == batch_size_limit or i == num_bins - 1:
                one_hot_batch = one_hot_encode(batch_sequences)
                
                # Slice into the H5 dataset using our tracking indices
                start_idx = batch_indices[0]
                end_idx = batch_indices[-1] + 1
                x_ds[start_idx:end_idx] = one_hot_batch
                
                # Reset batch containers
                batch_sequences = []
                batch_indices = []

            if i % 50000 == 0 and i > 0:
                print(f"Processed {i}/{num_bins} bins...")

    print("Success! Dataset generated successfully.")

if __name__ == "__main__":
    build_pipeline()

    print("\nVerifying H5 file structures and shapes:")
    with h5py.File(OUTPUT_H5, 'r') as hf:
        print(f"Keys in H5 file: {list(hf.keys())}")
        print(f"Shape of X (Input Data):  {hf['X'].shape}")
        print(f"Shape of y (Labels):      {hf['y'].shape}")
    
    
    y = np.loadtxt("data/bed_data/labels_matrix.txt", dtype=np.uint8)

    print("\nPlotting label distribution...")
    label_sums = np.sum(y, axis=0)  # sequences per track
    tracks = [f"Track {i+1}" for i in range(5)]

    plt.figure(figsize=(8,5))
    plt.bar(tracks, label_sums)
    plt.ylabel("Number of sequences")
    plt.title("Label distribution per track")
    plt.tight_layout()

    plt.savefig("label_distribution.png")
    print(f"Label distribution plot saved!")