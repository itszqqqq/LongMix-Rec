# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation
from unirank.pytorch.layers.tokenization import AutoSplitTokenizer


# [修改 1] padding 在归一化、FFN、残差后仍须为零；masked_fill 不会提升 BF16 dtype。
def _zero_padding(tokens, mask):
    if mask is None:
        return tokens
    if mask.shape != tokens.shape[:2]:
        raise ValueError("[LONGER] padding mask must match tokens [B, L].")
    valid = mask.to(device=tokens.device, dtype=torch.bool)
    return tokens.masked_fill(~valid.unsqueeze(-1), 0.0)


# [修改 1] 合并 padding/因果约束；显式处理没有合法 key 的行，避免全屏蔽 softmax。
def _masked_attention(attention, Q, K, V, scale, key_mask=None,
                      causal_mask=None):
    # 构造一个便于广播的 mask。支持组内 [B,Lq,D] 和多头 [B,H,Lq,Dh]；bool True 表示允许读取。
    allowed = torch.ones((Q.size(0),) + (1,) * (Q.dim() - 2) + (K.size(-2),),
                         dtype=torch.bool, device=Q.device)

    if key_mask is not None:
        if key_mask.shape != (Q.size(0), K.size(-2)):
            raise ValueError("[LONGER] key mask must match [B, KV_length].")
        allowed = allowed & key_mask.to(device=Q.device, dtype=torch.bool).reshape(allowed.shape)

    if causal_mask is not None:
        if causal_mask.shape != (Q.size(-2), K.size(-2)):
            raise ValueError("[LONGER] causal mask must match [Q_length, KV_length].")
        allowed = allowed & causal_mask.to(device=Q.device, dtype=torch.bool)
    
    has_key = allowed.any(dim=-1, keepdim=True)   # dim=-1 是 key 维；判断每个 query 有没有合法 key。
    first_key = torch.arange(K.size(-2), device=Q.device) == 0  # K.size(-2)取得 key 的数量
    # 无合法 key 时临时放行 key 0，计算后丢弃该行，并由调用方在 FFN 后再次清零。
    safe_allowed = allowed | (~has_key & first_key)
    output, _ = attention(Q, K, V, scale=scale, mask=safe_allowed, is_causal=False)  

    return output.masked_fill(~has_key, 0.0), has_key


class LONGER(MultiTaskModel):
    def __init__(self,
                 feature_map,    # feature_map 对象
                 model_id="LONGER",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="GeLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],    
                 embedding_dim=16,  # 每个类别字段的 embedding 维度
                 num_layers=3,     # self attention 层数
                 expansion_factor=4,  # FFN 中间层维度扩展倍数
                 num_heads=2,   # cross/self attention 的 head 数
                 num_tasks=6,  # 任务数
                 token_dim=256,   # 模型使用的 token 维度
                 num_ns_token=4,  # 非序列/global token 数
                 num_groups=4,  # 每组包含 4 个历史 token
                 query_size=10,   # [修改 3] 选取最近 n 个合并后的历史组作 query
                 net_dropout=0,
                 accumulation_steps=1,
                 max_len=100,
                 **kwargs):
        super(LONGER, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     **kwargs)

        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_ns_token = num_ns_token
        self.accumulation_steps = accumulation_steps
        self.max_len = max_len
        self.num_groups = num_groups
        self.query_size = query_size

        # [修改 3] 提前拒绝无效长度/组大小，避免除零及后续预测塔维度不匹配。
        if max_len <= 0 or num_groups <= 0:
            raise ValueError("[LONGER] max_len and num_groups must be positive.")
        if max_len % num_groups != 0:
            raise ValueError(
                f"[LONGER] sequence length Ls={max_len} is not divisible by num_groups={num_groups}. "
                f"Please truncate/pad in data pipeline or set compatible num_groups."
            )
        if not 1 <= query_size <= max_len // num_groups:
            raise ValueError(
                f"[LONGER] query_size={query_size} must be in [1, {max_len // num_groups}] "
                f"for max_len={max_len}, num_groups={num_groups}."
            )

        self.item_info_dim = 0
        self.non_item_dim = 0
        for feat, spec in self.feature_map.features.items():  # 遍历所有特征
            if feat in self.feature_map.labels: 
                continue
            if spec.get("type") == "meta":  # 如果特征类型是 meta，则跳过
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)   # 获取 embedding 维度
            if spec.get("source") in ["item", "action"]:   # 如果特征来源是 item 或 action，则将其 embedding 维度累加到 item_info_dim 中
                self.item_info_dim += emb_dim  # item_id, video_type, primary_tag, music_type, duration_bucket, action
            else:
                self.non_item_dim += emb_dim  # user id, 用户画像(26个特征), context(4个特征)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)  # 创建一个可训练的模块，遍历 feature_map，为各个需要 embedding 的字段创建对应的层

        self.unified_tokenizer_layer = AutoSplitTokenizer(
            input_dim=self.non_item_dim + self.item_info_dim,  # user context embedding + candidate item embedding
            token_dim=token_dim,
            num_tokens=num_ns_token
        )  

        # 新增：历史窗口的可学习绝对位置；在特征投影前相加。
        # 使用 Parameter，使其进入 dense optimizer；候选物品不加位置。
        self.history_position_embedding = nn.Parameter(torch.empty(max_len, self.item_info_dim))
        nn.init.normal_(self.history_position_embedding, std=0.02)

        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)   
        else:
            self.item_token_proj = nn.Identity()   

        # 创建 LONGERBlock 模块，包含 inner_trans, cross_causal_attention, self_causal_attention 
        self.unified_interaction_layers = LONGERBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            num_layers=num_layers,
            query_size=self.query_size,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations
        )
        # 创建一个 MLP_Block 列表，每个 MLP_Block 对应一个任务，用于将 token_dim * (num_ns_token + query_size) 维的输入映射到 1 维输出
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=token_dim * (num_ns_token + self.query_size),
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout)
            for _ in range(num_tasks)
        ])

        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])  # task 是一个列表，表示每个任务的类型，如 ["binary_classification", "regression", "multi_classification"]，每个任务使用不同的输出激活函数
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]  # task 是一个字符串，表示任务类型，如 "binary_classification" 所有任务使用相同的输出激活函数
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))   # 配置优化器和各任务损失；
        self.reset_parameters()   # 初始化网络权重
        self.model_to_device()   # 将模型移动到指定的设备上（CPU 或 GPU）

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)    # 每个字段输出：（batch_size, seq_len, embedding_dim），flatten_emb=True 表示将所有 embedding 拼接成一个向量
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)   # (batch_size, seq_len, item_info_dim)，其中 seq_len 是序列长度，item_info_dim 是所有 item 特征的 embedding 维度总和

        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, 0:-1, :]

        # s_tokens = self.item_token_proj(sequence_emb)
        length = sequence_emb.size(1)
        if length <= 0 or length > self.max_len:
            raise ValueError(f"[LONGER] history length {length} must be in [1, {self.max_len}].")
        if mask.shape != sequence_emb.shape[:2]:
            raise ValueError("[LONGER] mask must match history [B, Ls].")
        
        # 位置编码。与左 padding 对齐：短窗口使用位置表的末尾，最新行为固定在最后位置。
        positions = self.history_position_embedding[-length:].unsqueeze(0)
        s_tokens = self.item_token_proj(sequence_emb + positions)
        s_tokens = _zero_padding(s_tokens, mask)  # [修改 1] 去除位置/投影产生的 padding 值。

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)  # (batch_size, non_item_dim)，其中 non_item_dim 是所有非 item 特征的 embedding 维度总和
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)  # 与候选表示拼接
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)   # global token，(batch_size, num_ns_token, token_dim)

        # activation checkpoint: “用额外计算换显存”的训练技术。前向传播时少保存中间结果，反向传播需要时再重新计算。
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,  # 要执行的模块
            s_tokens,  # 第一个输入：历史序列 token，(batch_size, seq_len-1, token_dim)
            unified_tokens,  # 第二个输入：global token，(batch_size, num_ns_token, token_dim)
            mask   # 第三个输入：mask，(batch_size, seq_len-1)
        )

        bottom_output = unified_tokens.flatten(start_dim=1)  # (batch_size, (num_ns_token + query_size) * token_dim)，将 global token 和 query token 拼接成一个向量
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]  # 每个任务分别输出 (batch_size, 1) 的 logits
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]   # 对二分类任务分别应用 Sigmoid，得到 6 组 [B, 1] 概率

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]   # 按标签名存储预测结果，如 {"is_click_pred": [B, 1], "is_like_pred": [B, 1], ...}
        return return_dict


class LONGERBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_groups,
                 num_layers,
                 query_size,
                 dnn_activations='ReLU',
                 expansion_factor=4):
        super(LONGERBlock, self).__init__()
        self.num_layers = num_layers
        self.query_size = query_size
        # [修改 3] 独立使用 LONGERBlock 时也拒绝 q=0 和 K<=0。
        if num_groups <= 0 or query_size <= 0:
            raise ValueError("[LONGER] num_groups and query_size must be positive.")

        self.inner_trans_norm = nn.LayerNorm(input_dim)
        self.cross_attention_norm = nn.LayerNorm(input_dim)
        self.self_attention_norm = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])

        self.inner_trans = InnerTrans(
            input_dim=input_dim,
            num_groups=num_groups,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations
        )

        self.cross_causal_attention = CrossCausalAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations
        )

        self.self_attention_layers = nn.ModuleList([
            SelfCausalAttention(
                input_dim=input_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dnn_activations=dnn_activations
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens: B x Ls x D
        ns_tokens: B x Lns x D
        mask: B x Ls
        """
        # [修改 3] 按实际 padding 后的长度检查，支持构造后传入较短窗口的情况。
        length = s_tokens.size(1)
        group_size = self.inner_trans.num_groups
        if length <= 0 or length % group_size != 0:
            raise ValueError("[LONGER] history length must be positive and divisible by num_groups.")
        merged_length = length // group_size
        if not 1 <= self.query_size <= merged_length:
            raise ValueError(
                f"[LONGER] query_size={self.query_size} must be in [1, {merged_length}] "
                f"for runtime history length={length}, num_groups={group_size}."
            )

        # [修改 1] 原始 history mask -> merged mask -> query/key mask。
        if mask is None:
            mask = torch.ones(s_tokens.shape[:2], dtype=torch.bool, device=s_tokens.device)
        else:
            mask = mask.to(device=s_tokens.device, dtype=torch.bool)

        s_tokens = _zero_padding(s_tokens, mask)
        s_tokens = _zero_padding(self.inner_trans_norm(s_tokens), mask)
        
        merge_s_tokens = self.inner_trans(s_tokens, mask=mask)
        merged_mask = mask.reshape(mask.size(0), merged_length, group_size).any(dim=-1)
        
        sampled_s_tokens = merge_s_tokens[:, -self.query_size:, :]
        sampled_mask = merged_mask[:, -self.query_size:]
        global_mask = torch.ones(ns_tokens.shape[:2], dtype=torch.bool, device=ns_tokens.device)
        
        q_tokens = torch.cat([sampled_s_tokens, ns_tokens], dim=1)
        kv_tokens = torch.cat([merge_s_tokens, ns_tokens], dim=1)
        query_mask = torch.cat([sampled_mask, global_mask], dim=1)
        key_mask = torch.cat([merged_mask, global_mask], dim=1)

        # [修改 2] Q = [最近 q 个历史组, global]，KV = [全部历史组, global]。
        # 历史 query i 只读 j<=i 的历史组，不能读取包含候选信息的 global token；global query 可以读取全部有效输入
        # i 必须是原合并序列下标，不能直接对非方形 Q/KV 使用默认 tril。
        history_q_pos = torch.arange(merged_length - self.query_size, merged_length,
                                     device=s_tokens.device)
        key_pos = torch.arange(kv_tokens.size(1), device=s_tokens.device)
        history_allowed = key_pos.unsqueeze(0) <= history_q_pos.unsqueeze(1)
        global_allowed = torch.ones((ns_tokens.size(1), kv_tokens.size(1)),
                                    dtype=torch.bool, device=s_tokens.device)
        cross_causal_mask = torch.cat([history_allowed, global_allowed], dim=0)

        tokens = self.cross_causal_attention(
            self.cross_attention_norm(q_tokens),
            self.cross_attention_norm(kv_tokens),
            query_mask=query_mask, key_mask=key_mask,
            causal_mask=cross_causal_mask,
        ) + q_tokens

        tokens = _zero_padding(tokens, query_mask)
        for i in range(self.num_layers):
            tokens = self.self_attention_layers[i](
                self.self_attention_norm[i](tokens), mask=query_mask
            ) + tokens
            tokens = _zero_padding(tokens, query_mask)
        return tokens


class InnerTrans(nn.Module):
    def __init__(self, input_dim, num_groups=4, expansion_factor=4, dnn_activations='ReLU'):
        super(InnerTrans, self).__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups

        if num_groups <= 0:  # [修改 3] 独立调用时也检查组大小。
            raise ValueError("[LONGER] num_groups must be positive.")
        hidden_dim = input_dim * num_groups

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()  

        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),  
            get_activation(dnn_activations),
            nn.Linear(hidden_dim * expansion_factor, input_dim)  # 这里把合并 token 压回了 D 维
        )

    def forward(self, s_tokens, mask=None):  # [修改 1] 接收组内 padding mask。
        """
        s_tokens: B x Ls x D
        return:
            merge_s_tokens: B x Lms x D
        """
        B, Ls, D = s_tokens.shape   

        if Ls <= 0 or Ls % self.num_groups != 0:
            raise ValueError(
                f"[InnerTrans] sequence length Ls={Ls} is not divisible by num_groups={self.num_groups}."
            )

        Lms = Ls // self.num_groups  
        if mask is None:
            mask = torch.ones((B, Ls), dtype=torch.bool, device=s_tokens.device)
        else:
            mask = mask.to(device=s_tokens.device, dtype=torch.bool)
        s_tokens = _zero_padding(s_tokens, mask)
        group_mask = mask.reshape(B * Lms, self.num_groups)

        # Group adjacent tokens in chunks of num_groups
        merge_s_tokens = s_tokens.reshape(B, Lms, self.num_groups, D)   

        merge_s_tokens = merge_s_tokens.reshape(B * Lms, self.num_groups, D) # 合并到第一维不会让不同用户或不同组相互 attention。 attention 在每个小序列内部进行

        Q = self.W_q(merge_s_tokens)
        K = self.W_k(merge_s_tokens)
        V = self.W_v(merge_s_tokens)

        # [修改 1] padding key 不参与归一化，padding query 不进入后面的 KD 拼接。
        output, _ = _masked_attention(self.dot_attention, Q, K, V,
                                      scale=self.input_dim ** 0.5, key_mask=group_mask)
        output = _zero_padding(output, group_mask)

        output = output.reshape(B, Lms, self.num_groups * D)   
        merge_s_tokens = self.FFN(output)   # (B, Lms, D)
        merged_mask = group_mask.reshape(B, Lms, self.num_groups).any(dim=-1)
        merge_s_tokens = _zero_padding(merge_s_tokens, merged_mask)   # [修改 1] 清除全空组的 FFN bias。

        return merge_s_tokens


class CrossCausalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, expansion_factor=4, dnn_activations='ReLU'):
        super(CrossCausalAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()

        self.FFN = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            get_activation(dnn_activations),
            nn.Linear(input_dim * expansion_factor, input_dim)
        )

    def forward(self, q_tokens, kv_tokens, query_mask=None, key_mask=None,
                causal_mask=None):  # [修改 1/2] padding 与组级因果约束。
        """
        q_tokens: B x (Lq + Lns) x D
        kv_tokens: B x (Lms + Lns) x D
        """
        B, _, D = q_tokens.shape
        q_tokens = _zero_padding(q_tokens, query_mask)
        kv_tokens = _zero_padding(kv_tokens, key_mask)

        Q = self.W_q(q_tokens)
        K = self.W_k(kv_tokens)
        V = self.W_v(kv_tokens)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # [修改 2] 非方形 attention 不能猜测 query 的原时间位置，必须由调用方传入。
        if causal_mask is None:
            raise ValueError("[LONGER] CrossCausalAttention requires an explicit causal_mask.")
        output, has_key = _masked_attention(
            self.dot_attention, Q, K, V, scale=self.head_dim ** 0.5,
            key_mask=key_mask, causal_mask=causal_mask,
        )

        output = output.transpose(1, 2).contiguous().view(B, -1, D)
        output = self.FFN(output)

        # [修改 1] 无合法 key 的行在 FFN 后也必须为零；不能保留 FFN bias。
        output = output.masked_fill(~has_key.squeeze(1), 0.0)
        output = _zero_padding(output, query_mask)
        return output


class SelfCausalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, expansion_factor=4, dnn_activations='ReLU'):
        super(SelfCausalAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()

        self.FFN = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            get_activation(dnn_activations),
            nn.Linear(input_dim * expansion_factor, input_dim)
        )

    def forward(self, tokens, mask=None):  # [修改 1] padding mask 与原下三角约束合并。
        """
        tokens: B x (Lq + Lns) x D
        """
        B, L, D = tokens.shape
        tokens = _zero_padding(tokens, mask)

        Q = self.W_q(tokens)
        K = self.W_k(tokens)
        V = self.W_v(tokens)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # [修改 1] 显式合并 key padding 与因果 mask，避免无效 query 行产生伪输出。
        causal_mask = torch.ones((L, L), dtype=torch.bool, device=tokens.device).tril()
        output, _ = _masked_attention(
            self.dot_attention, Q, K, V, scale=self.head_dim ** 0.5,
            key_mask=mask, causal_mask=causal_mask,
        )

        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.FFN(output)
        output = _zero_padding(output, mask)
        return output
