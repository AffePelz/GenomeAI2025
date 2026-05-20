import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
from models import DeepSEA
from kmer_models import DeepSEA_Kmer

class GenomicDataset:
    def __init__(self, fasta_file=None, l=1000, k=2):
        self.fasta_file = fasta_file
        self.l = l
        self.k = k
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def read_fasta(self) -> str:
        n=20000000
        if self.fasta_file is None:
            np.random.seed(42)
            return "".join(np.random.choice(['A', 'C', 'G', 'T'], size=self.l))
            
        try:
            sequence = []
            with open(self.fasta_file, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        continue
                    sequence.append(line.strip().upper())
            return ''.join(sequence)[n : n + self.l]
        except (FileNotFoundError, TypeError):
            # Safe catch-all fallback if the file path is broken
            np.random.seed(42)
            return "".join(np.random.choice(['A', 'C', 'G', 'T'], size=self.l))

    @staticmethod
    def k_mers(sequence, k=2):
        return [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    
    """@staticmethod
    def k_mers_non_overlapping(sequence, k=2):
        return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k)]"""

    def k_mers_encoded(self, sequence):
        kmers = self.k_mers(sequence, self.k)

        encoded = []

        for kmer in kmers:
            try:
                idx = 0
                for ch in kmer:
                    idx = idx * 4 + self.base_map[ch]
                encoded.append(idx)
            except KeyError:
                continue
        
        return torch.tensor(encoded, dtype=torch.long)

def one_hot_encode(sequence):
    mapping = {
        "A": [1,0,0,0],
        "C": [0,1,0,0],
        "G": [0,0,1,0],
        "T": [0,0,0,1]
    }

    encoded = np.array([mapping[base] for base in sequence])

    return encoded

if __name__ == "__main__":
    K_MER = 3
    #fasta_file = "data/genome/hg38.fa"

    dataset = GenomicDataset(l=1000, k=K_MER)
    seq = dataset.read_fasta()

    # Encode k-mers
    x = dataset.k_mers_encoded(seq)
    x = x.unsqueeze(0)  # add batch dim -> (B, L)

    # Model
    model = DeepSEA_Kmer(k=K_MER)

    with torch.no_grad():
        out = model(x)

    logits = out["logits"]

    print("Logits shape:", logits.shape)
    print(logits)
    
    """encoded_sequence = one_hot_encode(seq)

    print(encoded_sequence.shape)
    # (1000, 4)


    # ---------------------------------------------------
    # Convert to tensor
    # ---------------------------------------------------

    input_tensor = torch.tensor(
        encoded_sequence,
        dtype=torch.float32
    )

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    print(input_tensor.shape)
    # (1, 1000, 4)


    # ---------------------------------------------------
    # Create model
    # ---------------------------------------------------

    model = DeepSEA()

    # Evaluation mode
    model.eval()


    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------

    with torch.no_grad():
        outputs = model(input_tensor)

    logits = outputs["logits"]

    print(logits.shape)
    # (1, 919)

    print(logits)"""