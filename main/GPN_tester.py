import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import random

# -----------------------------
# 1. Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FASTA_FILE = "data/genome/Homo_sapiens.chr22.dna.primary_assembly.fa"

SEQ_LEN = 512
BATCH_SIZE = 8
EPOCHS = 5

TOP_K = 500
MOTIF_WIDTH = 21

CHUNK_SIZE = 50_000   # IMPORTANT: prevents OOM

NUCLEOTIDES = ["A", "C", "G", "T"]
stoi = {c: i for i, c in enumerate(NUCLEOTIDES)}
MASK_TOKEN = 4
VOCAB_SIZE = 5

# -----------------------------
# 2. FASTA loader
# -----------------------------
def load_fasta(filename):
    seqs = []
    with open(filename) as f:
        s = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if s:
                    seqs.append(s)
                    s = ""
            else:
                s += line.upper()
        if s:
            seqs.append(s)
    return seqs

def clean(seq):
    return "".join(c for c in seq if c in NUCLEOTIDES)

def encode(seq):
    return torch.tensor([stoi[c] for c in seq], dtype=torch.long)

print("Loading genome...")
raw = load_fasta(FASTA_FILE)
genome = "".join(clean(s) for s in raw)
genome_tensor = encode(genome)

print("Genome length:", len(genome))

# -----------------------------
# 3. Model
# -----------------------------
class DNAModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d)

        self.conv = nn.Sequential(
            nn.Conv1d(d, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, d, 7, padding=3),
            nn.ReLU(),
        )

        self.fc = nn.Linear(d, 4)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.fc(x)

model = DNAModel().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 4. Training
# -----------------------------
print("Training...")

def sample_batch():
    batch = []
    for _ in range(BATCH_SIZE):
        i = random.randint(0, len(genome_tensor) - SEQ_LEN - 1)
        batch.append(genome_tensor[i:i+SEQ_LEN])
    return torch.stack(batch)

def mask(x):
    m = torch.rand_like(x.float()) < 0.15
    y = x.clone()
    x2 = x.clone()
    x2[m] = MASK_TOKEN
    return x2, y, m

model.train()

for ep in range(EPOCHS):
    x = sample_batch().to(DEVICE)
    x, y, m = mask(x)

    out = model(x)
    loss = F.cross_entropy(out[m], y[m])

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"Epoch {ep+1} loss {loss.item():.4f}")

# -----------------------------
# 5. SAFE genome scanning (NO OOM)
# -----------------------------
print("Scanning genome safely...")

model.eval()
scores = []

with torch.no_grad():
    for start in range(0, len(genome_tensor), CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, len(genome_tensor))
        chunk = genome_tensor[start:end].to(DEVICE).unsqueeze(0)

        logits = model(chunk)
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values[0].cpu()

        scores.append(conf)

scores = torch.cat(scores).numpy()

# -----------------------------
# 6. Top-k positions
# -----------------------------
top_idx = np.argsort(scores)[-TOP_K:]
top_pos = top_idx

# -----------------------------
# 7. Extract motifs
# -----------------------------
HALF = MOTIF_WIDTH // 2
seqs = []

for p in top_pos:
    if p - HALF < 0 or p + HALF >= len(genome):
        continue
    seqs.append(genome[p-HALF:p+HALF+1])

print("Motifs collected:", len(seqs))

# -----------------------------
# 8. PWM
# -----------------------------
pwm = np.zeros((len(seqs[0]), 4))

for s in seqs:
    for i, c in enumerate(s):
        pwm[i, stoi[c]] += 1

pwm /= len(seqs)

df = pd.DataFrame(pwm, columns=["A", "C", "G", "T"])

info = df * (np.log2(df + 1e-6) - np.log2(0.25))

#plt.figure(figsize=(12,4))
logomaker.Logo(info)
plt.title("DNA Motif (Safe Scan Version)")
plt.savefig("motif_safe_scan.png")
plt.show()