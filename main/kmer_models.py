import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import PreTrainedModel
from seq2fun.config import Seq2FunConfig
from typing import Optional, Dict, Tuple

class DeepSEA_Kmer(nn.Module):
    def __init__(self, k=2, n_targets=919, sequence_length=1000):
        super().__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.k = k
        self.n_targets = n_targets
        self.sequence_length = sequence_length
        
        vocab_size = 4 ** k

        # k-mer embedding (Outputs 1 feature dimension per token)
        self.embedding = nn.Embedding(vocab_size, 1)

        self.conv_net = nn.Sequential(
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
        # FIXED TYPO: Your original comment said "Input expected as (B, L, 4)" 
        # But for k-mers, input_ids MUST be integer tokens of shape (B, L_k)
        x = self.embedding(input_ids)        # (B, L_k, 1)
        x = x.transpose(1, 2)                # (B, 1, L_k)

        x = self.conv_net(x)                 # (B, 960, L_padded)
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

class DanQ_Kmer(nn.Module):
    def __init__(self, k=2, n_targets=919, sequence_length=1000, embedding_dim=4):
        super().__init__()

        self.k = k
        self.n_targets = n_targets
        self.sequence_length = sequence_length

        self.embedding = nn.Embedding(
            num_embeddings=4**k,
            embedding_dim=embedding_dim
        )

        # FIX A: Changed in_channels from 4 to embedding_dim
        self.conv_pool_drop_1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=320, kernel_size=26, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )

        self.bdlstm = nn.LSTM(
            input_size=320,
            hidden_size=320,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout_2 = nn.Dropout(0.5)

        # FIX B: Use a dummy pass to safely find the true sequence length 
        # without worrying about math formula mismatches
        with torch.no_grad():
            effective_input_len = sequence_length - k + 1  # Overlapping token length
            dummy_x = torch.zeros(1, embedding_dim, effective_input_len)
            dummy_conv = self.conv_pool_drop_1(dummy_x)
            pool_out_len = dummy_conv.shape[-1]
            
        lstm_out_dim = pool_out_len * 2 * 320

        self.dense_1 = nn.Sequential(
            nn.Linear(lstm_out_dim, 925),
            nn.ReLU()
        )
        self.dense_2 = nn.Linear(925, n_targets)
        self.register_buffer('pos_weight', None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        x = self.conv_pool_drop_1(x)
        x = x.transpose(1, 2)
        x, _ = self.bdlstm(x)
        x = self.dropout_2(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense_1(x)
        logits = self.dense_2(x)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=self.pos_weight, reduction="mean")
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss

        return outputs

class Flow_Attention_Kmer(nn.Module):
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
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        return torch.einsum("nhld,nhdm->nhlm", q, kv)

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
        
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                               (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
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


class DeepFormer_Kmer(nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000, k=2, embedding_dim=32):
        super().__init__()
        self.sequence_length = sequence_length
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        vocab_size = 4 ** k
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        self.conv_net1 = nn.Sequential(
            nn.Conv1d(embedding_dim, 320, kernel_size=conv_kernel_size),
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
        
        # FIX 1: Explicitly force the spatial sequence dimension to exactly 44 
        # before passing it to the attention block.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(44)
        
        # Now 44 is completely safe to hardcode here!
        self.attn_normal = Flow_Attention_Kmer(44, 44, 44, 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(44 * 960, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )
        
        self.register_buffer('pos_weight', None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        x = self.conv_net1(x)
        x = self.conv_net2(x)
        x = self.conv_net3(x)
        
        # FIX 2: Apply adaptive pooling here
        x = self.adaptive_pool(x)  # Output shape is guaranteed to be (B, 960, 44)
        
        x = self.attn_normal(x, x, x)
        x = x.view(x.size(0), 44 * 960)
        logits = self.classifier(x)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=self.pos_weight, reduction="mean")
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss
        return outputs

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_pool: int = 1,
        kernel_size: int = 9,
        dropout: float = 0.2,
        use_exp: bool = False,
        pool_type: str = "stride"  # "stride" or "maxpool"
    ):
        super().__init__()
        stride = n_pool if pool_type == "stride" else 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU() if not use_exp else torch.exp
        self.dropout = nn.Dropout(dropout)
        
        self.pool = None
        if pool_type == "maxpool" and n_pool > 1:
            self.pool = nn.MaxPool1d(kernel_size=n_pool, stride=n_pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x) if callable(self.activation) else self.activation(x)
        x = self.dropout(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class Seq2FunModel(PreTrainedModel):
    config_class = Seq2FunConfig
    base_model_prefix = "seq2fun"
    _supports_sdpa = True
    
    def __init__(self, config, k=None, embedding_dim=None):
        super().__init__(config)
        self.config = config
        
        # Pull values dynamically from config, fallback to explicit parameters if provided
        self.k = k if k is not None else getattr(config, "kmer_k", 2)
        self.embedding_dim = embedding_dim if embedding_dim is not None else getattr(config, "embedding_dim", 128)
        
        vocab_size = 4 ** self.k
        
        # Add the embedding layer 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        
        n_pools = [5, 5, 2]
        pool_scale = math.prod(n_pools)
        self.n_tokens = int(config.sequence_length // pool_scale)
        self.max_n_tokens = config.max_position_embeddings
        
        # CRITICAL CHANGE: Change 4 channels to self.embedding_dim
        self.conv1 = ConvBlock(self.embedding_dim, 320, n_pools[0], 15, use_exp=True, pool_type="maxpool")
        self.conv2 = ConvBlock(320, 320, n_pools[1], 5, use_exp=False, pool_type="maxpool")
        self.conv3 = ConvBlock(320, config.hidden_size, n_pools[2], 5, use_exp=False, pool_type="maxpool")
        
        self.transformer = BertEncoder(config)
        
        self.projection = nn.Linear(config.hidden_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(256 * self.max_n_tokens, 256),
            nn.GELU(),
            nn.Linear(256, config.num_tracks),
        )
        
        self.register_buffer('pos_weight', None)
    
    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # input_ids shape from dataset: (B, L)
        
        # 1. Map token IDs to vectors: (B, L) -> (B, L, Embedding_Dim)
        x = self.embedding(input_ids)
        
        # 2. Reshape for Conv1D layout: (B, L, Embedding_Dim) -> (B, Embedding_Dim, L)
        x = x.transpose(1, 2)
        
        # 3. Conv tower (Now receives Embedding_Dim channels perfectly!)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)  # (B, L', H)
        
        # Pad before transformer
        x, attn_mask = self.pad_center(x)  

        # Run BertEncoder
        encoder_outputs = self.transformer(hidden_states=x)
        x = encoder_outputs.last_hidden_state  
        
        # Projection + classification
        x = self.projection(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
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
        
    def pad_center(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = tensor.shape
        target_length = self.max_n_tokens
    
        if L >= target_length:
            padded = tensor[:, :target_length, :]
            # float mask: 0 for real tokens
            mask = torch.zeros(B, target_length, target_length, dtype=tensor.dtype, device=tensor.device)
            return padded, mask
    
        pad_total = target_length - L
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
    
        padded_tensor = F.pad(tensor, (0, 0, pad_left, pad_right), mode="constant", value=0)
    
        # float mask: -1e9 for padding, 0 for real tokens
        mask = torch.full((target_length, target_length), fill_value=-1e4, dtype=tensor.dtype, device=tensor.device)
        # Only allow attention to real tokens along S dimension
        mask[:, pad_left:pad_left + L] = 0.0
    
        return padded_tensor, mask
    
class BinTEN1(PreTrainedModel):
    config_class = Seq2FunConfig
    base_model_prefix = "BinTEN"
    _supports_sdpa = True
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 1. Dynamically read k and embedding dimensions from the updated configuration object
        self.k = getattr(config, "kmer_k", 2)
        self.embedding_dim = getattr(config, "embedding_dim", 128)
        
        vocab_size = 4 ** self.k
        
        # 2. Add Token Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        
        n_pools = [2, 2, 2, 2]
        pool_scale = math.prod(n_pools)
        
        # Note: If your sequence length shortens due to overlapping k-mers (L - k + 1), 
        # ensure your config.sequence_length reflects that value so the linear classifier shape maps correctly.
        self.n_tokens = int(config.sequence_length // pool_scale)
        
        # 3. CRITICAL CHANGE: Swap 4 input channels to self.embedding_dim
        self.conv1 = ConvBlock(self.embedding_dim, 320, n_pools[0], 15, use_exp=True, pool_type="maxpool")
        self.conv2 = ConvBlock(320, 480, n_pools[1], 9, use_exp=False, pool_type="maxpool")
        self.conv3 = ConvBlock(480, 640, n_pools[2], 9, use_exp=False, pool_type="maxpool")
        self.conv4 = ConvBlock(640, config.hidden_size, n_pools[3], 9, use_exp=False, pool_type="maxpool")
        
        self.transformer = BertEncoder(config)
        
        self.projection = nn.Linear(config.hidden_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(256 * self.n_tokens, 256),
            nn.GELU(),
            nn.Linear(256, config.num_tracks),
        )
        
        self.register_buffer('pos_weight', None)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # input_ids expected shape: (Batch_Size, Sequence_Length) containing integer k-mer IDs
        
        # 1. Convert token IDs to continuous vectors: (B, L) -> (B, L, Embedding_Dim)
        x = self.embedding(input_ids)
        
        # 2. Re-orient spatial dimensions for Conv1D processing: (B, Embedding_Dim, L)
        x = x.transpose(1, 2)
        
        # 3. Process through Conv feature extractor tower
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.transpose(1, 2)  # Revert shape layout to: (B, L', H)
        
        # 4. Contextual sequence modeling via Transformer Encoder
        encoder_outputs = self.transformer(hidden_states=x)
        x = encoder_outputs.last_hidden_state  # (B, n_tokens, H)
        
        # 5. Dimensionality reduction projection
        x = self.projection(x)
        x = self.dropout(x)
        
        # Flatten temporal dimensions into classification representations
        x = x.reshape(x.shape[0], -1)
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

class MSAEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, aux_features_vocab_size: int, embedding_size: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.aux_features_vocab_size = aux_features_vocab_size
        self.embedding_size = embedding_size

    def forward(self, human_feature, aux_features=None):
        if human_feature is None:
            raise Exception("human_feature is missing")

        # Start with DNA token embeddings
        dna_embeddings = F.one_hot(human_feature, num_classes=self.vocab_size).float()
        embeddings_list = [dna_embeddings]

        # Add aux_features if provided
        if aux_features is not None:
            if self.aux_features_vocab_size is not None:
                aux_features = (
                    F.one_hot(
                        aux_features.long(), num_classes=self.aux_features_vocab_size
                    )
                    .reshape(human_feature.shape[0], human_feature.shape[1], -1)
                    .float()
                )
            embeddings_list.append(aux_features)

        # Concatenate all embeddings contiguously
        res = torch.cat(embeddings_list, dim=-1)

        # Pad to embedding_size if needed
        current_size = res.shape[-1]
        if current_size < self.embedding_size:
            padding = torch.zeros(
                res.shape[0],
                res.shape[1],
                self.embedding_size - current_size,
                device=res.device,
                dtype=res.dtype,
            )
            res = torch.cat([res, padding], dim=-1)
        return res