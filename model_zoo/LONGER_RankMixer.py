
"""Full LONGER output -> masked mean -> feature concat -> Chunk -> RankMixer.

Reuses the mask/causal-corrected LONGERBlock in the UniRank checkout.
Both stages are trained end to end by the inherited multi-task loss.
"""

import torch
from torch import nn

from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.layers.tokenization import AutoSplitTokenizer, ChunkTokenizer
from .LONGER import LONGERBlock, _zero_padding
from .RankMixer import RankMixerBlock


class LongerEncoder(nn.Module):
    """Encode embedded history; return all q+G outputs and their validity mask."""

    def __init__(
        self, item_info_dim, non_item_dim, token_dim=256, max_len=100,
        num_groups=4, query_size=10, num_global_tokens=4,
        num_heads=2, num_layers=1, expansion_factor=4,
        dnn_activations="ReLU",
    ):
        super().__init__()
        if min(item_info_dim, non_item_dim, token_dim, max_len, num_groups,
               num_global_tokens, num_heads, expansion_factor) <= 0:
            raise ValueError("Encoder dimensions, groups, heads and expansion must be positive.")
        if max_len % num_groups or token_dim % num_heads:
            raise ValueError("max_len % num_groups and token_dim % num_heads must be zero.")
        if not 1 <= query_size <= max_len // num_groups or num_layers < 0:
            raise ValueError("Invalid merged query count or encoder layer count.")
        
        self.item_info_dim = item_info_dim
        self.context_target_dim = non_item_dim + item_info_dim
        self.max_len = max_len
        self.num_groups = num_groups
        self.query_size = query_size
        self.num_global_tokens = num_global_tokens

        # Keep the current LONGER position convention: add before projection,
        # right-align the position table for left-padded, possibly shorter windows.
        self.history_position_embedding = nn.Parameter(
            torch.empty(max_len, item_info_dim)
        )
        nn.init.normal_(self.history_position_embedding, std=0.02)
        self.item_token_proj = (
            nn.Linear(item_info_dim, token_dim)
            if item_info_dim != token_dim else nn.Identity()
        )
        # First tokenizer: user/context + candidate -> LONGER global queries.
        self.global_tokenizer = AutoSplitTokenizer(
            input_dim=self.context_target_dim,
            token_dim=token_dim,
            num_tokens=num_global_tokens,
        )
        self.block = LONGERBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            num_layers=num_layers,
            query_size=query_size,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations,
        )

    def forward(self, history_emb, context_target_emb, history_mask):
        if history_emb.ndim != 3 or history_emb.size(-1) != self.item_info_dim:
            raise ValueError("history_emb must have shape [B,L,item_info_dim].")
        batch_size, length, _ = history_emb.shape
        if not 0 < length <= self.max_len or length % self.num_groups:
            raise ValueError("History window must be within max_len and divisible by num_groups.")
        if self.query_size > length // self.num_groups:
            raise ValueError("query_size exceeds the runtime number of merged groups.")
        if context_target_emb.shape != (batch_size, self.context_target_dim):
            raise ValueError("context_target_emb must have shape [B,non_item_dim+item_info_dim].")
        if history_mask.shape != history_emb.shape[:2]:
            raise ValueError("history_mask must have shape [B,L].")
        history_mask = history_mask.to(device=history_emb.device, dtype=torch.bool)

        positions = self.history_position_embedding[-length:].unsqueeze(0)
        history_tokens = self.item_token_proj(history_emb + positions)
        history_tokens = _zero_padding(history_tokens, history_mask)
        global_tokens = self.global_tokenizer(context_target_emb)
        encoded = self.block(history_tokens, global_tokens, history_mask)

        # This mirrors the current LONGERBlock's query order exactly:
        # [last q merged-history queries, G global queries].
        merged_mask = history_mask.reshape(
            batch_size, length // self.num_groups, self.num_groups
        ).any(dim=-1)
        global_mask = torch.ones(
            (batch_size, self.num_global_tokens),
            dtype=torch.bool, device=encoded.device,
        )
        output_mask = torch.cat([
            merged_mask[:, -self.query_size:], global_mask,
        ], dim=1)
        if encoded.shape[:2] != output_mask.shape:
            raise ValueError("LONGER output order/length no longer matches the readout mask.")
        return encoded, output_mask


class MaskedMeanReadout(nn.Module):
    """Read all valid encoded queries, using FP32 for BF16/FP16 accumulation."""

    def forward(self, encoded, output_mask):
        if encoded.ndim != 3 or output_mask.shape != encoded.shape[:2]:
            raise ValueError("Expected encoded [B,N,D] and output_mask [B,N].")

        valid = output_mask.to(device=encoded.device, dtype=torch.bool)
        clean = encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
        accumulator = clean.float() if clean.dtype in (torch.float16, torch.bfloat16) else clean
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(accumulator.dtype)

        return (accumulator.sum(dim=1) / count).to(encoded.dtype)


class LONGER_RankMixer(MultiTaskModel):
    def __init__(
        self, feature_map, model_id="LONGER_RankMixer",
        task="binary_classification", gpu=-1,
        embedding_dim=16, token_dim=256, max_len=100,
        num_groups=4, query_size=10, num_global_tokens=4,
        num_rank_tokens=8, num_heads=2,
        longer_num_layers=1, longer_expansion_factor=4,
        rankmixer_num_layers=1, rankmixer_expansion_factor=2,
        dnn_activations="ReLU", tower_hidden_units=(128, 64),
        tower_activations="ReLU", num_tasks=None, net_dropout=0,
        accumulation_steps=1, **kwargs,
    ):
        # Reject legacy aliases that would silently obscure the two stages.
        legacy_keys = {"num_layers", "expansion_factor", "num_ns_token", "tokenizer_type"} & kwargs.keys()
        if legacy_keys:
            raise ValueError(
                f"Use the explicit encoder/global/rank settings instead of {sorted(legacy_keys)}."
            )
        actual_tasks = len(feature_map.labels)
        num_tasks = actual_tasks if num_tasks is None else num_tasks
        if actual_tasks == 0 or num_tasks != actual_tasks:
            raise ValueError("num_tasks must equal len(feature_map.labels) > 0.")
        if num_rank_tokens <= 0 or rankmixer_num_layers < 0 or rankmixer_expansion_factor <= 0:
            raise ValueError("Invalid rank token count, layer count or expansion.")
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive.")
        loss = kwargs.get("loss", "binary_crossentropy")
        if isinstance(loss, list) and len(loss) != num_tasks:
            raise ValueError("loss list must have one entry per label.")
        
        super().__init__(
            feature_map, model_id=model_id, task=task,
            num_tasks=num_tasks, gpu=gpu, **kwargs,
        )

        self.max_len = max_len
        self.token_dim = token_dim
        self.num_global_tokens = num_global_tokens
        self.num_rank_tokens = num_rank_tokens
        self.accumulation_steps = accumulation_steps
        self.item_info_dim = 0
        self.non_item_dim = 0

        for name, spec in feature_map.features.items():
            if name in feature_map.labels or spec.get("type") == "meta":
                continue
            width = spec.get("emb_output_dim", spec.get("embedding_dim", embedding_dim))
            if spec.get("source") in ("item", "action"):
                self.item_info_dim += width
            else:
                self.non_item_dim += width

        # One shared embedding module for history, candidate and user/context.
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.longer_encoder = LongerEncoder(
            item_info_dim=self.item_info_dim,
            non_item_dim=self.non_item_dim,
            token_dim=token_dim, max_len=max_len,
            num_groups=num_groups, query_size=query_size,
            num_global_tokens=num_global_tokens, num_heads=num_heads,
            num_layers=longer_num_layers,
            expansion_factor=longer_expansion_factor,
            dnn_activations=dnn_activations,
        )
        self.readout = MaskedMeanReadout()
        self.rank_input_dim = self.non_item_dim + self.item_info_dim + token_dim
        # Second tokenizer: raw user/context + candidate + LONGER summary.
        self.rank_tokenizer = ChunkTokenizer(
            input_dim=self.rank_input_dim,
            token_dim=token_dim,
            num_tokens=num_rank_tokens,
        )
        self.rankmixer = (
            RankMixerBlock(
                input_dim=token_dim, num_ns_token=num_rank_tokens,
                num_layers=rankmixer_num_layers,
                expand=rankmixer_expansion_factor, net_dropout=net_dropout,
            ) if rankmixer_num_layers > 0 else nn.Identity()
        )
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=token_dim, output_dim=1,
                hidden_units=list(tower_hidden_units),
                hidden_activations=tower_activations,
                output_activation=None, dropout_rates=net_dropout,
            ) for _ in range(num_tasks)
        ])

        # Optimizers see every component, including both tokenizers.
        self.compile(
            kwargs.get("dense_optimizer"), loss,
            kwargs.get("dense_learning_rate"),
        )
        self.reset_parameters()
        self.model_to_device()

    def encode_inputs(self, inputs):
        batch_dict, item_dict, history_mask = self.get_inputs(inputs)
        batch_size = history_mask.size(0)

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.reshape(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]
        history_emb = item_seq_emb[:, :-1, :]

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        context_target_emb = torch.cat([user_context_emb, target_emb], dim=-1)

        encoded, output_mask = self.activation_checkpoint(
            self.longer_encoder, history_emb, context_target_emb, history_mask,
        )
        return user_context_emb, target_emb, encoded, output_mask

    def forward(self, inputs):
        user_context_emb, target_emb, encoded, output_mask = self.encode_inputs(inputs)
        h_long = self.readout(encoded, output_mask)  # [B,D], candidate-aware.

        features = torch.cat([user_context_emb, target_emb, h_long], dim=-1)
        tokens = self.rank_tokenizer(features)  # [B,T_rank,D], all fixed valid slots.
        
        mixed = self.activation_checkpoint(self.rankmixer, tokens)
        bottom = mixed.mean(dim=1)

        return {
            f"{label}_pred": self.output_activation[i](self.tower[i](bottom))
            for i, label in enumerate(self.feature_map.labels)
        }
