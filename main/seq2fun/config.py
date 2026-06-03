from transformers import BertConfig

class Seq2FunConfig(BertConfig):
    model_type = "seq2fun"
    ALLOWED_SEQUENCE_LENGTHS = [1000, 5000, 10000, 20000, 30000, 50000]

    def __init__(
        self,
        num_tracks=1,
        sequence_length=1000,
        num_hidden_layers=1,
        num_attention_heads=8,
        hidden_size=512,
        position_embedding_type="absolute",
        max_position_embeddings=1000,
        k=2,               # <--- Added: General k length default
        embedding_dim=128,      # <--- Added: Embedding dimension default
        **kwargs
    ):
        if sequence_length not in self.ALLOWED_SEQUENCE_LENGTHS:
            raise ValueError(
                f"sequence_length must be one of {self.ALLOWED_SEQUENCE_LENGTHS}, "
                f"but got {sequence_length}"
            )

        # Standard BertConfig args
        super().__init__(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            **kwargs
        )

        # Custom arguments
        self.num_tracks = num_tracks
        self.sequence_length = sequence_length
        self.position_embedding_type = position_embedding_type
        
        # K-mer specific arguments
        self.k = k
        self.embedding_dim = embedding_dim

        # Internal defaults like original BERT
        self._attn_implementation = 'sdpa'

class MSA2FunConfig(BertConfig):
    model_type = "msa2fun"

    def __init__(
        self,
        # MSA2Fun specific
        num_tracks=423,
        sequence_length: int = 1200,
        vocab_size: int = 5,
        embedding_size: int = 512,
        n_aux_features: int = 5 * 89,
        n_msa: int = 90,
        aux_features_vocab_size: int = 5,
        reverse_complement_prob: float = 0.5,
        classifier_hidden_size: int = 256,
        n_pools: list[int] = [3, 2, 1],
        conv_hidden: int = 512,
        dropout: float = 0.5,
        task_type: str = "binary",  # "binary" or "quantitative"
        counts_loss_weight: float = 1.0,
        # BertConfig args
        num_hidden_layers: int = 1,
        num_attention_heads: int = 8,
        hidden_size: int = 512,
        max_position_embeddings: int = 200,
        **kwargs,
    ):
        kwargs.pop("mask_token_id", None)

        super().__init__(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )

        # MSA2Fun specific
        self.num_tracks = num_tracks
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_aux_features = n_aux_features
        self.n_msa = n_msa
        self.aux_features_vocab_size = aux_features_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_pools = n_pools
        self.classifier_hidden_size = classifier_hidden_size
        self.conv_hidden = conv_hidden
        self.dropout = dropout
        self.reverse_complement_prob = reverse_complement_prob
        self.task_type = task_type
        self.counts_loss_weight = counts_loss_weight
        self._attn_implementation = 'sdpa'