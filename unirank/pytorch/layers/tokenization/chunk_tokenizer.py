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

import math
import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.layers.blocks import MLP_Block


class ChunkTokenizer(nn.Module):
    '''Tokenizer that splits a flattened feature vector into chunks and applies an MLP to each chunk.
    把每条样本的完整特征向量连续切成若干块，每块通过独立的投影网络，生成固定维度的 token'''
    def __init__(self, input_dim, token_dim, num_tokens, activation="SiLU",
                 layer_norm=True):
        super(ChunkTokenizer, self).__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        
        self.input_dim = input_dim
        self.num_tokens = num_tokens  # tokenizer 输出 token 的数量
        self.chunk_dim = int(math.ceil(float(input_dim) / float(num_tokens)))  # 每个 chunk 的维度
        self.padded_input_dim = self.chunk_dim * num_tokens  # 填充后的输入维度
        self.pad_dim = self.padded_input_dim - input_dim  # 需要补 padding 的数量
        self.token_mlps = nn.ModuleList([
            MLP_Block(
                input_dim=self.chunk_dim,
                hidden_units=[token_dim],
                hidden_activations=activation,
                layer_norm=layer_norm,
            )
            for _ in range(num_tokens)
        ])

    def forward(self, feature_embeddings):
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1) # 将输入展平为二维张量
        if feature_embeddings.size(-1) != self.input_dim:
            raise ValueError(
                "ChunkTokenizer expects input_dim={}, got {}.".format(
                    self.input_dim, feature_embeddings.size(-1)
                )
            )
        if self.pad_dim > 0:
            feature_embeddings = F.pad(feature_embeddings, (0, self.pad_dim))   # 填充 padding
        chunks = feature_embeddings.view(-1, self.num_tokens, self.chunk_dim)   # 将 flatten 后的输入张量切分为 num_tokens 个 chunk，每个 chunk 的维度为 chunk_dim
        
        return torch.stack([    # 用 stack 组成 token 张量
            mlp(chunks[:, i, :])   # 对每个 chunk 应用对应的 MLP
            for i, mlp in enumerate(self.token_mlps)   
        ], dim=1)   #  [B,num_tokens,token_dim]
