import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from typing import Optional, Dict
import math
import os
import json


class DeepSEA(nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.n_targets = n_targets
        self.sequence_length = sequence_length

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            ((
                (sequence_length - reduce_by) / pool_kernel_size
            //1 - reduce_by) / pool_kernel_size
            //1 - reduce_by)
        )

        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )
        
        # Register pos_weight as buffer (will move to GPU with model)
        self.register_buffer('pos_weight', None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Input expected as (B, L, 4) -> transpose to (B, 4, L)
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            input_ids = input_ids.transpose(1, 2)

        out = self.conv_net(input_ids)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        logits = self.classifier(reshape_out)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), 
                    pos_weight=self.pos_weight, 
                    reduction="mean"
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss

        return outputs


class DanQ(nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        self.n_targets = n_targets
        self.sequence_length = sequence_length

        # Conv + Pool + Drop
        self.conv_pool_drop_1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )

        # Bidirectional LSTM
        self.bdlstm = nn.LSTM(
            input_size=320,
            hidden_size=320,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout_2 = nn.Dropout(0.5)

        # Compute flattened size after conv + pool
        conv_out_len = (sequence_length - 26 + 1)  # Conv1d output
        pool_out_len = conv_out_len // 13          # MaxPool1d output
        lstm_out_dim = pool_out_len * 2 * 320      # bidirectional LSTM

        # Dense layers
        self.dense_1 = nn.Sequential(
            nn.Linear(lstm_out_dim, 925),
            nn.ReLU()
        )
        self.dense_2 = nn.Linear(925, n_targets)
        
        # Register pos_weight as buffer
        self.register_buffer('pos_weight', None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # Transpose to (B, 4, L) if needed
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            input_ids = input_ids.transpose(1, 2)

        x = self.conv_pool_drop_1(input_ids)  # (B, 320, L')
        x = x.transpose(1, 2)                 # (B, L', 320)
        x, _ = self.bdlstm(x)                 # (B, L', 640)
        x = self.dropout_2(x)
        x = x.reshape(x.size(0), -1)          # Flatten
        x = self.dense_1(x)
        logits = self.dense_2(x)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), 
                    pos_weight=self.pos_weight, 
                    reduction="mean"
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss
        return outputs


class Flow_Attention(nn.Module):
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=5e-4):
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = einsum("nhld,nhlm->nhdm", k, v)
        return einsum("nhld,nhdm->nhlm", q, kv)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
      
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        
        sink_incoming = 1.0 / (einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        conserved_sink = einsum("nhld,nhd->nhl", queries + self.eps,
                               (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = einsum("nhld,nhd->nhl", keys + self.eps,
                                 (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)
       
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],
                              keys,
                              values * source_competition[:, :, :, None])
             * sink_allocation[:, :, :, None]).transpose(1, 2)
       
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


class DeepFormer(nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        self.sequence_length = sequence_length  # dummy for HF Trainer
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        self.conv_net1 = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320)
        )
        self.conv_net2 = nn.Sequential(
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2)
        )
        self.conv_net3 = nn.Sequential(
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2)
        )
        self.attn_normal = Flow_Attention(44, 44, 44, 4)
        self.classifier = nn.Sequential(
            nn.Linear(44 * 960, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )
        
        # Register pos_weight as buffer
        self.register_buffer('pos_weight', None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # transpose if input is (B, L, 4)
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            x = input_ids.transpose(1, 2)
        else:
            x = input_ids

        x = self.conv_net1(x)
        x = self.conv_net2(x)
        x = self.conv_net3(x)
        x = self.attn_normal(x, x, x)
        x = x.view(x.size(0), 44 * 960)
        logits = self.classifier(x)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), 
                    pos_weight=self.pos_weight, 
                    reduction="mean"
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss
        return outputs