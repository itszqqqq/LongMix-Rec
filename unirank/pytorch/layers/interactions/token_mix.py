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
import torch.nn.functional as F

class MultiHeadTokenMixing(nn.Module):
    def __init__(self, input_dim, num_token):
        super(MultiHeadTokenMixing, self).__init__()
        self.num_token = num_token # = num_heads
        self.input_dim = input_dim
        self.pad_dim = (-input_dim) % num_token   # 补 padding 的维度，使得 input_dim 可以被 num_token 整除
        self.padded_input_dim = input_dim + self.pad_dim
        self.head_dim = self.padded_input_dim // num_token   # num_token = num_heads

        valid_mask = torch.arange(self.padded_input_dim) < input_dim
        valid_mask = valid_mask.expand(num_token, -1)  # valid_mask 用来标记哪些坐标来自原始输入、哪些坐标是为了分 head 而补入的 padding。
        
        valid_mask = valid_mask.reshape(
            num_token, num_token, self.head_dim
        ).transpose(0, 1).reshape(-1)
        
        self.register_buffer(
            "valid_index",
            valid_mask.nonzero(as_tuple=False).flatten(),
            persistent=False,
        )

    def forward(self, x):  # x: [B, T, D]
        if self.pad_dim:
            x = F.pad(x, (0, self.pad_dim))
        
        heads = torch.tensor_split(x, self.num_token, dim=-1)   # heads[i] 的维度为 [B, T, head_dim]
        mixed = torch.stack(heads, dim=1).flatten(start_dim=2)   # [B, H, T, head_dim] -> [B, H, T * head_dim]
        mixed = mixed.flatten(start_dim=1).index_select(1, self.valid_index)   # 先把每个样本的所有坐标展平，再删除 mixing 后散落在不同位置的 padding 坐标
        # [B, H, T*head_dim] -> flatten -> [B, H*T*head_dim] -> index_select -> [B, T*D]

        return mixed.reshape(x.size(0), self.num_token, self.input_dim)   # (B, T, D)
