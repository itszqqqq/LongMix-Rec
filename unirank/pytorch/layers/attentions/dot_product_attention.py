# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2018. pengshuang@Github for ScaledDotProductAttention.
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
import torch.nn.functional as F
from torch import nn
from unirank.pytorch.torch_utils import get_activation


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with configurable score activation."""

    SUPPORTED_ACTIVATIONS = {
        "softmax": "SoftMax",
        "none": "None",
        "relu": "ReLU",
        "softplus": "SoftPlus",
        "silu": "SiLU",
        "gelu": "GeLU",
        "sigmoid": "Sigmoid",
        "mish": "Mish",
    }

    def __init__(self, dropout_rate=0., attention_activation_type="SoftMax"):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate
        activation_key = (
            "none"
            if attention_activation_type is None
            else str(attention_activation_type).strip().lower()
        )
        if activation_key not in self.SUPPORTED_ACTIVATIONS:
            supported = ", ".join(self.SUPPORTED_ACTIVATIONS.values())
            raise ValueError(
                "Unsupported attention_activation_type={!r}. Expected one of: {}"
                .format(attention_activation_type, supported)
            )
        self.attention_activation_type = self.SUPPORTED_ACTIVATIONS[activation_key]
        self.attention_activation = (
            None
            if self.attention_activation_type == "SoftMax"
            else get_activation(self.attention_activation_type)
        )

    def forward(self, Q, K, V, scale=None, mask=None, is_causal=False):
        # mask: 0 for masked positions
        if self.attention_activation_type == "SoftMax":
            if mask is not None:
                mask = mask.bool()
            sdpa_scale = None if scale is None else 1.0 / scale
            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
                scale=sdpa_scale,
            )
            return output, None

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        if scale is not None:
            attention_scores = attention_scores / scale
        attention_weights = attention_scores
        if self.attention_activation is not None:
            attention_weights = self.attention_activation(attention_scores)

        valid_mask = mask.bool() if mask is not None else None

        if is_causal:
            query_length, key_length = Q.size(-2), K.size(-2)
            causal_mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=Q.device,
            ).tril()
            valid_mask = causal_mask if valid_mask is None else valid_mask & causal_mask

        if valid_mask is not None:
            attention_weights = attention_weights.masked_fill(~valid_mask, 0.0)

        attention_weights = F.dropout(
            attention_weights,
            p=self.dropout_rate,
            training=self.training,
        )
        output = torch.matmul(attention_weights, V)
        return output, None

