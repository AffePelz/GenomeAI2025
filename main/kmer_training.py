import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List
from kmer_models import DeepSEA_Kmer
import random

# =====================================================================
# 1. DATA PREPARATION: Fixed-Vocab Hash-based Tokenization
# =====================================================================

def dna_to_one_hot(sequence: str) -> torch.Tensor:
    """Converts a DNA string into a One-Hot encoded Tensor (L, 4)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = [mapping.get(base, 0) for base in sequence.upper()]
    one_hot = F.one_hot(torch.tensor(encoding), num_classes=4)
    return one_hot.float()

class DNAPairDataset(Dataset):
    """
    Optimized Dataset using a rolling-style hash function to map large k-mers 
    to a fixed token space without storing explicit combination arrays.
    """
    def __init__(self, sequences: List[str], k: int, vocab_size: int = 1000000):
        self.k = k
        self.sequences = [seq.upper() for seq in sequences]
        # Bounding vocabulary to 1M max to protect GPU/System RAM for k > 11
        self.vocab_size = min(4 ** k, vocab_size)
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def _kmer_to_hash_idx(self, kmer: str) -> int:
        """Computes a numerical index for the k-mer safely bounded by a modulo."""
        val = 0
        for base in kmer:
            val = (val * 4 + self.base_map.get(base, 0)) % self.vocab_size
        return val

    def _seq_to_kmer_indices(self, sequence: str) -> torch.Tensor:
        num_kmers = len(sequence) - self.k + 1
        if num_kmers <= 0:
            raise ValueError(f"Sequence length ({len(sequence)}) is smaller than k ({self.k})")
            
        indices = []
        for i in range(num_kmers):
            kmer = sequence[i:i+self.k]
            indices.append(self._kmer_to_hash_idx(kmer))
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        one_hot = dna_to_one_hot(seq)
        kmer_indices = self._seq_to_kmer_indices(seq)
        return one_hot, kmer_indices

if __name__ == "__main__":
    # 1. Gather all prime numbers up to 50
    primes_up_to_50 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_targets = 919
    
    # Sequence length needs to be comfortably larger than our highest k (47)
    # so that Conv1d downsampling operations still find enough spatial data.
    sequence_length = 1000 
    steps = 5
    
    print(f"--- Starting All-Prime Sweep Up To 50 ---")
    print(f"Primes to test: {primes_up_to_50}")
    print(f"Sequence length per sample: {sequence_length} bp\n")

    # 2. Generate Static Data & Targets (Ensures identical benchmarks)
    random.seed(42)
    torch.manual_seed(42)
    bases = ['A', 'C', 'G', 'T']
    raw_dna_data = ["".join(random.choices(bases, k=sequence_length)) for _ in range(4)]
    mock_labels = torch.randint(0, 2, (len(raw_dna_data), n_targets))
    
    results = {}

    # 3. Loop through every prime architecture
    for k in primes_up_to_50:
        # Reset seeds inside the loop so weight distribution properties match initially
        torch.manual_seed(42)
        
        # Prepare pipeline data
        dataset = DNAPairDataset(raw_dna_data, k=k, vocab_size=2*k)
        dataloader = DataLoader(dataset, batch_size=len(raw_dna_data), shuffle=False)
        _, kmer_batch = next(iter(dataloader))
        
        # Initialize Architecture & Optimizer
        model = DeepSEA_Kmer(k=k, embedding_dim=320, n_targets=n_targets)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        # Mini Training Loop
        history = []
        for step in range(steps):
            optimizer.zero_grad()
            outputs = model(kmer_batch, mock_labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            
        results[k] = history
        print(f"Successfully swept k={k:<2} | Effective Vocab Size: {dataset.vocab_size:<9,} | Final Step Loss: {history[-1]:.6f}")

    # Display Comprehensive Summary Table
    print(f"\n=========================================================")
    print(f"     FINAL COMPREHENSIVE BENCHMARK (ALL PRIMES UP TO 50)  ")
    print(f"=========================================================")
    print(f" Prime k  | Initial Loss (Step 1) | Final Loss (Step {steps})")
    print(f"---------------------------------------------------------")
    for k in primes_up_to_50:
        initial_loss = results[k][0]
        final_loss = results[k][-1]
        print(f"  k = {k:<2}   |        {initial_loss:.6f}       |      {final_loss:.6f}")
    print(f"=========================================================")