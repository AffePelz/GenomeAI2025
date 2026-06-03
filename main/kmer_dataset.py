import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
from seq2fun.config import Seq2FunConfig
from kmer_models import DeepSEA_Kmer, DanQ_Kmer, Flow_Attention_Kmer, DeepFormer_Kmer, Seq2FunModel, BinTEN1

class GenomicDataset:
    def __init__(self, fasta_file=None, l=1000, k=2):
        self.fasta_file = fasta_file
        self.l = l
        self.k = k
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def read_fasta(self) -> str:
        if self.fasta_file is None:
            np.random.seed(42)
            return "".join(np.random.choice(['A', 'C', 'G', 'T'], size=self.l))
            
        try:
            n = 20000000
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
    
    @staticmethod
    def k_mers_non_overlapping(sequence, k=2):
        return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k)]

    def k_mers_encoded(self, sequence, kmer_method=k_mers):
        kmers = kmer_method(sequence, self.k)

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

    @staticmethod
    def one_hot_encode(sequence):
        onehot_encoding = {
            "A": [1,0,0,0],
            "C": [0,1,0,0],
            "G": [0,0,1,0],
            "T": [0,0,0,1]
        }

        encoded = np.array([onehot_encoding[base] for base in sequence])

        return encoded

if __name__ == "__main__":
    K_MER = 5
    L = 1000

    # Loading DNA sequence
    dataset = GenomicDataset(fasta_file=None, l=L, k=K_MER)
    raw_sequence = dataset.read_fasta()

    # Encoding the sequence into k-mers
    current_method = GenomicDataset.k_mers
    encoded_sequence = dataset.k_mers_encoded(raw_sequence, kmer_method=current_method)

    # Converting to tensor and adding batch dimension (Sequence_Length) -> (Batch_Size, Sequence_Length)
    input_tensor = encoded_sequence.unsqueeze(0)

    input_ids_batch = torch.stack([encoded_sequence, encoded_sequence], dim=0)
    mock_labels = torch.randint(0, 2, (2, 3)).float()

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Sample tokens: {input_tensor}")

    #model = DeepFormer_Kmer(n_targets=919, sequence_length=1000, k=K_MER, embedding_dim=32)
    #model = DanQ_Kmer(k=K_MER, n_targets=919, sequence_length=1000, embedding_dim=16)
    #model = DeepSEA_Kmer(k=K_MER, n_targets=919, sequence_length=1000)

    config = Seq2FunConfig(
        sequence_length=1000,  # Adjust for k-mer length
        kmer_k=K_MER, 
        embedding_dim=64, 
        hidden_size=128, 
        num_hidden_layers=5,
        num_tracks=919  # Match your dummy_labels shape!
    )   

    #model = Seq2FunModel(config)
    model = BinTEN1(config)
    model.eval() # Set to evaluation mode

    # 4. Generate dummy targets/labels for the loss check
    dummy_labels = torch.randint(0, 2, (1, 919)).float()

    # 5. Run the forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, labels=dummy_labels)

    # 6. Inspect the results
    print("\n--- Model Output ---")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss value:  {outputs['loss'].item()}")