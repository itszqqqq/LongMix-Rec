
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
        self,
        item_info_dim,
        non_item_dim,
        token_dim=256,
        max_len=100,
        num_groups=4,
        query_size=10,
        num_global_tokens=4,
        num_heads=2,
        num_layers=1,
        expansion_factor=4,
        dnn_activations="ReLU",
    ):
        super().__init__()

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

        # self.item_token_proj = (
        #     nn.Linear(item_info_dim, token_dim)
        #     if item_info_dim != token_dim else nn.Identity()
        # )
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # First tokenizer: user/context + candidate -> LONGER global queries.
        self.global_tokenizer = AutoSplitTokenizer(
            input_dim=self.context_target_dim,
            token_dim=token_dim,
            num_tokens=num_global_tokens,
        )
        self.longerblock = LONGERBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            num_layers=num_layers,
            query_size=query_size,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations,
        )

    def forward(self, history_emb, context_target_emb, history_mask):
        batch_size, length, _ = history_emb.shape  # shape:(B, seq_len, item_info_dim)

        history_mask = history_mask.to(device=history_emb.device, dtype=torch.bool)
        positions = self.history_position_embedding[-length:].unsqueeze(0)

        history_tokens = self.item_token_proj(history_emb + positions)  # [B,max_len,item_info_dim] → [B,max_len,token_dim]
        history_tokens = _zero_padding(history_tokens, history_mask)

        global_tokens = self.global_tokenizer(context_target_emb)
        encoded = self.longerblock(history_tokens, global_tokens, history_mask)
        # [最近 q 个合并历史 query 的输出 | G 个 global query 的输出]

        # This mirrors the current LONGERBlock's query order exactly:
        # [last q merged-history queries, G global queries].
        merged_mask = history_mask.reshape(
            batch_size, length // self.num_groups, self.num_groups
        ).any(dim=-1)
        # [B,L] -> reshape -> [B,L/K,K] -> any(dim=-1) -> [B,L/K]
        # .any(dim=-1) 表示：组内至少有一个有效位置，这个合并组就有效

        global_mask = torch.ones(
            (batch_size, self.num_global_tokens),
            dtype=torch.bool, device=encoded.device,
        )   #  global_mask：[B, num_global_tokens]，值全为 True

        output_mask = torch.cat([
            merged_mask[:, -self.query_size:], global_mask,
        ], dim=1)   # [最近 q 个合并组的有效性 | num_global_tokens 个 True]

        return encoded, output_mask


class MaskedMeanReadout(nn.Module):
    """Read all valid encoded queries, using FP32 for BF16/FP16 accumulation."""

    def forward(self, encoded, output_mask):
        valid = output_mask.to(device=encoded.device, dtype=torch.bool)
        clean = encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
        accumulator = clean.float() if clean.dtype in (torch.float16, torch.bfloat16) else clean  # 如果输入为 FP16 或 BF16，先转为 FP32，再进行求和与除法，减少低精度累计误差
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(accumulator.dtype)  # 计算每条样本的有效 token 数

        return (accumulator.sum(dim=1) / count).to(encoded.dtype)    # 先沿 token 维求和 [B,N,D] → [B,D]，再除以每条样本自己的有效 token 数
        #  .to(encoded.dtype) 将结果恢复为输入精度，例如 FP32 计算后转回 BF16，便于接入后续网络


class LONGER_RankMixer(MultiTaskModel):
    def __init__(
        self,
        feature_map,
        model_id="LONGER_RankMixer",
        task="binary_classification",
        gpu=-1,
        embedding_dim=16,
        token_dim=256,
        max_len=100,
        num_groups=4,
        query_size=10,
        num_global_tokens=4,
        num_rank_tokens=8,
        num_heads=2,
        longer_num_layers=1,
        longer_expansion_factor=4,
        rankmixer_num_layers=1,
        rankmixer_expansion_factor=2,
        dnn_activations="ReLU",
        tower_hidden_units=(128, 64),
        tower_activations="ReLU",
        num_tasks=None, net_dropout=0,
        accumulation_steps=1,
        **kwargs,
    ):
        num_tasks = len(feature_map.labels) if num_tasks is None else num_tasks

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

        for feat, spec in feature_map.features.items():
            if feat in feature_map.labels or spec.get("type") == "meta":
                continue
            emb_dim = spec.get("emb_output_dim", spec.get("embedding_dim", embedding_dim))
            if spec.get("source") in ("item", "action"):
                self.item_info_dim += emb_dim
            else:
                self.non_item_dim += emb_dim

        # One shared embedding module for history, candidate and user/context.
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.longer_encoder = LongerEncoder(
            item_info_dim=self.item_info_dim,
            non_item_dim=self.non_item_dim,
            token_dim=token_dim,
            max_len=max_len,
            num_groups=num_groups,
            query_size=query_size,
            num_global_tokens=num_global_tokens,
            num_heads=num_heads,
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
                input_dim=token_dim,
                num_ns_token=num_rank_tokens,
                num_layers=rankmixer_num_layers,
                expand=rankmixer_expansion_factor,
                net_dropout=net_dropout,
            ) if rankmixer_num_layers > 0 else nn.Identity()
        )
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=token_dim,
                output_dim=1,
                hidden_units=list(tower_hidden_units),
                hidden_activations=tower_activations,
                output_activation=None, dropout_rates=net_dropout,
            ) for _ in range(num_tasks)
        ])

        # Optimizers see every component, including both tokenizers.
        self.compile(
            kwargs.get("dense_optimizer"), kwargs.get("loss", "binary_crossentropy"),
            kwargs.get("dense_learning_rate"),
        )
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, history_mask = self.get_inputs(inputs)
        batch_size = history_mask.size(0)
        # item_dict:{"key": [history_items..., target_item]}
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.reshape(batch_size, -1, self.item_info_dim) # (B, T+1, item_info_dim)，其中 T 是历史 item 的数量

        target_emb = item_seq_emb[:, -1, :]
        history_emb = item_seq_emb[:, :-1, :]

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        context_target_emb = torch.cat([user_context_emb, target_emb], dim=-1)

        encoded, output_mask = self.activation_checkpoint(
            self.longer_encoder, history_emb, context_target_emb, history_mask,
        )
        h_long = self.readout(encoded, output_mask)  # [B,D], candidate-aware.

        features = torch.cat([user_context_emb, target_emb, h_long], dim=-1)
        tokens = self.rank_tokenizer(features)  # [B,T_rank,D], all fixed valid slots.

        mixed = self.activation_checkpoint(self.rankmixer, tokens)
        bottom = mixed.mean(dim=1)

        return {
            f"{label}_pred": self.output_activation[i](self.tower[i](bottom))
            for i, label in enumerate(self.feature_map.labels)
        }
