import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class DeepSEA_Kmer(nn.Module):
    def __init__(self, k=2, n_targets=919, sequence_length=1000):
        super().__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.k = k
        self.n_targets = n_targets
        self.sequence_length = sequence_length
        
        vocab_size = 4 ** k

        # k-mer embedding
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, 1)

        self.conv_net = nn.Sequential(
            #nn.Conv1d(embedding_dim, 320, kernel_size=conv_kernel_size),
            nn.Conv1d(1, 320, kernel_size=conv_kernel_size),
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
        # Input expected as (B, L, 4) -> transpose to (B, 4, L)
        x = self.embedding(input_ids)     # (B, L, D)
        x = x.transpose(1, 2)             # (B, D, L)

        x = self.conv_net(x)              # (B, 960, L')
        x = self.global_pool(x).squeeze(-1)  # (B, 960)

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