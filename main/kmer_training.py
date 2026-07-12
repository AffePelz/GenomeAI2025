"""import os
import sys
import time
import numpy as np
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. TOKENIZER & DATASET (Fixed with Hashing)
# ==========================================
class GenomicDataset:
    #Handles DNA sequence tokenization using a memory-safe hashing trick.
    #Instead of creating a 4^k dictionary, it hashes k-mers into a fixed vocabulary.
    def __init__(self, k: int, vocab_size: int = 250000):
        self.k = k
        self.vocab_size = vocab_size

    def tokenize_sequence(self, seq: str) -> List[int]:
        seq = seq.upper()
        tokens = []
        # Extract sliding window k-mers
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            # Use Python's built-in hash bounded by vocab_size
            # The addition of a salt string ensures consistency within the run
            token_id = abs(hash(kmer)) % self.vocab_size
            tokens.append(token_id)
            
        # Handle edge case where sequence is shorter than k
        if not tokens:
            tokens = [0] * (len(seq))
            
        return tokens

def one_hot_encode(seq: str) -> np.ndarray:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            encoding[i, mapping[base]] = 1.0
    return encoding

def process_fasta_dual_modes(fasta_path: Optional[str], k: int, vocab_size: int, window_size=1000, stride=2):
    tokenizer = GenomicDataset(k=k, vocab_size=vocab_size)
    
    # Check if a valid FASTA path is provided and exists
    if fasta_path is not None and os.path.exists(fasta_path):
        print(f"Reading {fasta_path}...")
        try:
            from Bio import SeqIO
            record = SeqIO.read(fasta_path, "fasta")
            sequence = str(record.seq)
            print(f"Loaded sequence length: {len(sequence)} bp")
        except ImportError:
            print("BioPython not installed. Falling back to synthetic generation.")
            sequence = "".join(np.random.choice(['A', 'C', 'G', 'T'], size=5000))
    else:
        print(f"FASTA file not found or set to None. Generating synthetic sequence...")
        np.random.seed(42)
        sequence = "".join(np.random.choice(['A', 'C', 'G', 'T'], size=5000))

    seq_len = len(sequence)
    one_hot_windows = []
    kmer_windows = []
    coordinates = []

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        sub_seq = sequence[start:end]
        
        if sub_seq.upper().count('N') > (window_size * 0.05): 
            continue

        # Format 1: One-Hot Matrix
        one_hot_windows.append(one_hot_encode(sub_seq))
        
        # Format 2: K-mer structural tokens
        kmer_windows.append(tokenizer.tokenize_sequence(sub_seq))
        coordinates.append((start, end))
        
    return (
        torch.tensor(np.array(one_hot_windows), dtype=torch.float32), 
        torch.tensor(np.array(kmer_windows), dtype=torch.long),
        coordinates
    )

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================
class DeepSEA(nn.Module):
    # Standard Baseline One-Hot DeepSEA Model
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            nn.Conv1d(320, 480, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            nn.Conv1d(480, 960, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(960, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # input shape: (B, L, 4) -> Transpose to (B, 4, L)
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)
        
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        return outputs


class DeepSEA_Kmer(nn.Module):
    # Memory-Safe K-mer DeepSEA Model with Dynamic Vocabulary Caps
    def __init__(self, vocab_size=250000, embedding_dim=64, n_targets=919):
        super().__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_targets = n_targets

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(960, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )

        self.register_buffer("pos_weight", None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.pos_weight = pos_weight
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)          # (B, L_k, embedding_dim)
        x = x.transpose(1, 2)                  # (B, embedding_dim, L_k)

        x = self.conv_net(x)                   
        x = self.global_pool(x).squeeze(-1)    

        logits = self.classifier(x)            
        outputs = {"logits": logits}

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                pos_weight=self.pos_weight if self.pos_weight is not None else None
            )
            outputs["loss"] = loss

        return outputs

# ==========================================
# 3. EVALUATION PIPELINE
# ==========================================
def evaluate_model(model, input_tensors, true_labels, batch_size, device):
    all_predictions = []
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(input_tensors), batch_size):
            batch_inputs = input_tensors[i : i + batch_size].to(device)
            batch_labels = true_labels[i : i + batch_size].to(device)
            
            outputs = model(batch_inputs, labels=batch_labels)
            
            total_loss += outputs["loss"].item()
            num_batches += 1
            
            probabilities = torch.sigmoid(outputs["logits"])
            all_predictions.append(probabilities.cpu().numpy())
            
    execution_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    predictions = np.vstack(all_predictions)
    
    return avg_loss, predictions, execution_time

# ==========================================
# 4. MAIN BENCHMARK EXECUTION
# ==========================================
def main():
    fasta_file = None  # Change to path if needed (e.g. "data/genome/chr22.fa")
    
    # Notice: k=13 works perfectly fine now without blowing up memory!
    k_value = 47
    vocab_cap = 250000  # Safe dimension bound for local hardware
    n_targets = 919
    batch_size = 16    # Lowered slightly to stay safe on low VRAM GPUs
    
    # 1. Load data
    oh_inputs, kmer_inputs, coords = process_fasta_dual_modes(
        fasta_file, k=k_value, vocab_size=vocab_cap, window_size=1000, stride=50
    )
    num_windows = oh_inputs.shape[0]
    print(f"Successfully processed {num_windows} genomic windows.\n")

    if num_windows == 0:
        print("No windows generated. Try reducing stride or extending sequence length.")
        return

    # 2. Synthesize Ground Truth
    true_labels = torch.randint(0, 2, (num_windows, n_targets)).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}\n")

    # 3. Benchmark One-Hot Model
    print("--- Evaluating Mode 1: One-Hot Encoding ---")
    model_oh = DeepSEA(n_targets=n_targets, sequence_length=1000).to(device)
    model_oh.eval()
    loss_oh, preds_oh, time_oh = evaluate_model(model_oh, oh_inputs, true_labels, batch_size, device)
    print(f"One-Hot Model Shape: {oh_inputs.shape}")
    print(f"Loss: {loss_oh:.5f} | Time: {time_oh:.3f}s")

    # 4. Benchmark K-mer Model
    print("\n--- Evaluating Mode 2: K-mer Encoding ---")
    model_kmer = DeepSEA_Kmer(vocab_size=vocab_cap, embedding_dim=128, n_targets=n_targets).to(device)
    model_kmer.eval()
    
    loss_kmer, preds_kmer, time_kmer = evaluate_model(model_kmer, kmer_inputs, true_labels, batch_size, device)
    print(f"K-mer Model Shape:   {kmer_inputs.shape}")
    print(f"Loss: {loss_kmer:.5f} | Time: {time_kmer:.3f}s")

    # 5. Summary Analysis
    print("\n================== COMPARISON SUMMARY ==================")
    print(f"{'Metric':<20} | {'One-Hot Model':<15} | {'K-mer Model':<15}")
    print("-" * 58)
    print(f"{'Tensor Data Shape':<20} | {str(list(oh_inputs.shape)):<15} | {str(list(kmer_inputs.shape)):<15}")
    print(f"{'Average BCE Loss':<20} | {loss_oh:<15.5f} | {loss_kmer:<15.5f}")
    print(f"{'Inference Speed':<20} | {f'{time_oh:.3f}s':<15} | {f'{time_kmer:.3f}s':<15}")
    
    mae_diff = np.mean(np.abs(preds_oh - preds_kmer))
    print(f"{'Preds Mean Abs Diff':<20} | {mae_diff:<33.5f}")
    print("========================================================")

if __name__ == "__main__":
    main()"""

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
        dataset = DNAPairDataset(raw_dna_data, k=k, vocab_size=1000000)
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