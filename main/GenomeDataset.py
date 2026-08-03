import os
import h5py
import torch
import pysam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybedtools import BedTool, cleanup
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Set, Tuple

# -------------------------------------------------------------------
# 1. Path Management
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FASTA_PATH = os.path.join(DATA_DIR, "FASTA", "hg38.fa")  # Update to your FASTA filename if needed
H5_PATH = os.path.join(DATA_DIR, "dataset.h5")

BED_FILES = [
    os.path.join(DATA_DIR, "BED", "ENCFF052RRA.bed"),
    os.path.join(DATA_DIR, "BED", "ENCFF053BLB.bed"),
    os.path.join(DATA_DIR, "BED", "ENCFF057CRD.bed"),
    os.path.join(DATA_DIR, "BED", "ENCFF060JHQ.bed"),
    os.path.join(DATA_DIR, "BED", "ENCFF078AMJ.bed")
]


def normalize_chrom(chrom: str) -> str:
    """Standardizes chromosome names (e.g., '22' -> 'chr22')."""
    chrom = str(chrom).strip().lower()
    if chrom in ("mt", "chrm", "mitochondria"):
        return "chrm"
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    return chrom

# -------------------------------------------------------------------
# 2. Sequence Extraction & Annotation Classes
# -------------------------------------------------------------------
class Genome:
    def __init__(self, path: str, include_chroms: Optional[Set[str]] = None):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"FASTA file not found at: {path}")

        self.include_chroms = (
            {normalize_chrom(c) for c in include_chroms} if include_chroms else None
        )
        self._genome: Dict[str, str] = self._load_fasta(path)

    def _load_fasta(self, path: str) -> Dict[str, str]:
        fasta = pysam.FastaFile(path)
        genome = {}
        try:
            for raw_chrom in fasta.references:
                norm = normalize_chrom(raw_chrom)
                if self.include_chroms is not None and norm not in self.include_chroms:
                    continue
                genome[norm] = fasta.fetch(raw_chrom)
        finally:
            fasta.close()
        return genome

    def get_binned_windows(
        self,
        chrom: str,
        bin_size: int = 200,
        flank_size: int = 400
    ) -> Tuple[List[Tuple[str, int, int]], List[str]]:
        norm_chrom = normalize_chrom(chrom)
        if norm_chrom not in self._genome:
            raise KeyError(f"Chromosome '{chrom}' not found.")

        chrom_seq = self._genome[norm_chrom]
        chrom_len = len(chrom_seq)
        target_window_len = bin_size + (2 * flank_size)  # 1,000 bp

        coords = []
        windows = []

        for bin_start in range(0, chrom_len, bin_size):
            bin_end = min(bin_start + bin_size, chrom_len)
            coords.append((norm_chrom, bin_start, bin_end))

            win_start = bin_start - flank_size
            win_end = bin_end + flank_size

            left_pad = 0
            if win_start < 0:
                left_pad = abs(win_start)
                win_start = 0

            if win_end > chrom_len:
                win_end = chrom_len

            chunk = chrom_seq[win_start:win_end]
            window_seq = ("N" * left_pad) + chunk

            if len(window_seq) < target_window_len:
                right_pad = target_window_len - len(window_seq)
                window_seq = window_seq + ("N" * right_pad)

            windows.append(window_seq)

        return coords, windows


class PeakAnnotator:
    def __init__(self, bed_paths: List[str]):
        self.bed_paths = bed_paths
        self.num_tracks = len(bed_paths)
        self.bed_tools: List[BedTool] = []

        for path in self.bed_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"BED file missing at: {path}")

            df = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1, 2], dtype={0: str})
            df[0] = df[0].apply(normalize_chrom)
            bed_str = df.to_csv(sep='\t', header=False, index=False)
            bt = BedTool(bed_str, from_string=True).sort()
            self.bed_tools.append(bt)

    def annotate_bins(
        self, 
        coords: List[Tuple[str, int, int]], 
        min_overlap_bp: int = 100
    ) -> np.ndarray:
        if not coords:
            return np.zeros((0, self.num_tracks), dtype=np.int8)

        bed_str = "\n".join(
            f"{normalize_chrom(chrom)}\t{start}\t{end}\t{i}" 
            for i, (chrom, start, end) in enumerate(coords)
        )
        bins_bt = BedTool(bed_str, from_string=True)
        labels = np.zeros((len(coords), self.num_tracks), dtype=np.int8)

        for track_idx, bed_tool in enumerate(self.bed_tools):
            intersection = bins_bt.intersect(bed_tool, wo=True)
            for feature in intersection:
                bin_idx = int(feature[3])
                overlap_bp = int(feature[-1])
                if overlap_bp >= min_overlap_bp:
                    labels[bin_idx, track_idx] = 1

        cleanup()
        return labels

# -------------------------------------------------------------------
# Plotting & Execution Logic
# -------------------------------------------------------------------
def plot_bed_track_distribution(
    labels: np.ndarray, 
    bed_paths: List[str], 
    output_filename: str = "bed_track_distribution.png"
):
    # Generates and saves a clean bar chart showing positive bins per BED track.
    num_bins, num_tracks = labels.shape
    pos_counts = np.sum(labels == 1, axis=0)
    percentages = (pos_counts / num_bins) * 100
    
    # Extract readable accession names (e.g., 'ENCFF052RRA')
    track_labels = [os.path.basename(p).replace(".bed", "") for p in bed_paths]

    # Print terminal output
    print("\n" + "=" * 60)
    print(f"TOTAL BINS EVALUATED: {num_bins:,}")
    print("=" * 60)
    for i in range(num_tracks):
        print(f"Track {i+1} [{track_labels[i]}]: {pos_counts[i]:10,} positive bins ({percentages[i]:5.2f}%)")

    # Plot configuration
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        track_labels, 
        pos_counts, 
        color="#2b5c8f", 
        edgecolor="#1a3857", 
        linewidth=1.2, 
        width=0.55
    )

    # Annotate bar tops with exact count and percentage
    for bar, count, pct in zip(bars, pos_counts, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{count:,}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", 
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
    
    # Give overhead room for annotations
    ax.set_ylim(0, max(pos_counts) * 1.18 if max(pos_counts) > 0 else 10)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"\nPlot saved to '{output_filename}'!")

# -------------------------------------------------------------------
# 3. Builder Function
# -------------------------------------------------------------------
def build_h5_dataset(
    fasta_path: str,
    bed_files: List[str],
    target_chroms: Set[str],
    output_h5: str
):
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    
    print(f"1. Reading sequence from '{fasta_path}' for {target_chroms}...")
    genome = Genome(fasta_path, include_chroms=target_chroms)

    all_coords = []
    all_windows = []
    
    for chrom in target_chroms:
        coords, windows = genome.get_binned_windows(chrom, bin_size=200, flank_size=400)
        all_coords.extend(coords)
        all_windows.extend(windows)

    num_samples = len(all_coords)
    print(f"2. Annotating {num_samples:,} genomic bins against {len(bed_files)} BED tracks...")
    annotator = PeakAnnotator(bed_files)
    labels = annotator.annotate_bins(all_coords, min_overlap_bp=100)

    print(f"3. Writing output HDF5 to '{output_h5}'...")
    ascii_sequences = np.frombuffer("".join(all_windows).encode("ascii"), dtype=np.uint8).reshape(num_samples, 1000)

    with h5py.File(output_h5, 'w') as hf:
        hf.create_dataset(
            'sequences', 
            data=ascii_sequences, 
            dtype=np.uint8, 
            chunks=(256, 1000), 
            compression="gzip"
        )
        hf.create_dataset(
            'labels', 
            data=labels, 
            dtype=np.int8, 
            chunks=(256, len(bed_files)), 
            compression="gzip"
        )

    print("Success! HDF5 file created.")


# -------------------------------------------------------------------
# Task 4: One-Hot Encoding & Pytorch Dataset
# -------------------------------------------------------------------
def fast_one_hot_encode(seq_bytes: np.ndarray) -> torch.Tensor:
    one_hot = np.zeros((len(seq_bytes), 4), dtype=np.float32)
    one_hot[(seq_bytes == 65) | (seq_bytes == 97), 0] = 1.0   # A / a
    one_hot[(seq_bytes == 67) | (seq_bytes == 99), 1] = 1.0   # C / c
    one_hot[(seq_bytes == 71) | (seq_bytes == 103), 2] = 1.0  # G / g
    one_hot[(seq_bytes == 84) | (seq_bytes == 116), 3] = 1.0  # T / t
    return torch.from_numpy(one_hot)


class GenomicDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._h5_file = None
        
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Cannot open HDF5 dataset. File missing at: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['sequences'])

    def _open_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        self._open_h5()
        seq_ascii = self._h5_file['sequences'][idx]
        y_label = self._h5_file['labels'][idx]

        x_tensor = fast_one_hot_encode(seq_ascii)
        y_tensor = torch.from_numpy(y_label).float()
        return x_tensor, y_tensor


# -------------------------------------------------------------------
# 5. Execution Pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate 200 bp bins for chr22
    genome = Genome("data/FASTA/chr22.fa", include_chroms={"chr22"})
    coords, windows = genome.get_binned_windows("chr22", bin_size=200, flank_size=400)

    # 2. Annotate bins using normalized PeakAnnotator
    annotator = PeakAnnotator(BED_FILES)
    labels = annotator.annotate_bins(coords, min_overlap_bp=100)

    # 3. Render and save plot
    plot_bed_track_distribution(labels, BED_FILES, output_filename="label_distribution.png")
    # Check if HDF5 dataset exists; build it if missing
    if not os.path.exists(H5_PATH):
        print(f"HDF5 file not found at '{H5_PATH}'. Generating dataset now...")
        build_h5_dataset(
            fasta_path=FASTA_PATH,
            bed_files=BED_FILES,
            target_chroms={"chr22"},
            output_h5=H5_PATH
        )
    else:
        print(f"Found existing HDF5 dataset at '{H5_PATH}'.")

    # Load dataset with PyTorch
    dataset = GenomicDataset(H5_PATH)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    print(f"\nTotal dataset size: {len(dataset):,} binned windows")

    for X_batch, y_batch in dataloader:
        print("\n--- Verification Batch ---")
        print(f"X batch shape: {X_batch.shape}")  # torch.Size([64, 1000, 4])
        print(f"y batch shape: {y_batch.shape}")  # torch.Size([64, 5])
        break