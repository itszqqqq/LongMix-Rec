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

class PerTokenFeedForward(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 expand=2,
                 net_dropout=0.0):
        super(PerTokenFeedForward, self).__init__()
        self.input_dim = input_dim    
        self.num_token = num_token
        self.hidden_dim = expand * input_dim
        self.W1 = nn.Parameter(torch.empty(self.num_token, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_token, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_token, self.hidden_dim, self.input_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_token, self.input_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.W1)
        # nn.init.xavier_uniform_(self.W2)

        for t in range(self.num_token):
            nn.init.xavier_uniform_(self.W1[t])  
            nn.init.xavier_uniform_(self.W2[t])  
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor):  # x: [B, T, D]
        h = torch.einsum("btd,tdk->btk", x, self.W1) + self.b1   
        h = self.act(h)
        h = self.dropout(h)
        y = torch.einsum("btk,tkd->btd", h, self.W2) + self.b2
        return y

class PerTokenSwiGLU(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 expand=2,
                 net_dropout=0.0):
        super(PerTokenSwiGLU, self).__init__()
        self.input_dim = input_dim
        self.num_token = num_token
        self.hidden_dim = expand * input_dim
        self.W1 = nn.Parameter(torch.empty(self.num_token, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_token, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_token, self.input_dim, self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_token, self.hidden_dim))
        self.W3 = nn.Parameter(torch.empty(self.num_token, self.hidden_dim, self.input_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor):  # x: [B, T, D]
        h1 = torch.einsum("btd,tdk->btk", x, self.W1) + self.b1
        h1 = self.act(h1)
        h1 = self.dropout(h1)
        h2 = torch.einsum("btd,tdk->btk", x, self.W2) + self.b2
        y = torch.einsum("btk,tkd->btd", h1 * h2, self.W3)
        return y
    

class SwiGLU(nn.Module):
    def __init__(self, input_dim, expand=4, net_dropout=0.0):
        super(SwiGLU, self).__init__()
        hidden_dim = int(input_dim * expand)
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

    def forward(self, x): # x: [B, T, D]
        y = self.act(self.W1(x)) * self.W2(x)
        y = self.dropout(y)
        y = self.W3(y)
        return y

class SparseMoELayer(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 expand=2,
                 num_experts=4,
                 net_dropout=0.0):
        super(SparseMoELayer, self).__init__()
        self.input_dim = input_dim          # D
        self.num_token = num_token        # T
        self.hidden_dim = expand * input_dim  # kD
        self.num_experts = num_experts      # E

        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

        self.W1 = nn.Parameter(torch.empty(self.num_token, self.num_experts, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_token, self.num_experts, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_token, self.num_experts, self.hidden_dim, self.input_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_token, self.num_experts, self.input_dim))

        self.hidden_act = nn.GELU()
        self.gate_act = nn.ReLU()

        self.router = nn.Parameter(torch.empty(self.num_ns_token, self.num_experts, self.input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.xavier_uniform_(self.router)

    def _ffn_all_experts(self, x: torch.Tensor):
        """
        x: [B, T, D] -> expert_out: [B, T, E, D]
        """
        h = torch.einsum("btd,tedk->btek", x, self.W1) + self.b1  # [B, T, E, kD]
        h = self.hidden_act(h)
        h = self.dropout(h)
        expert_out = torch.einsum("btek,tekd->bted", h, self.W2) + self.b2  # [B, T, E, D]
        return expert_out

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        assert T == self.num_ns_token and D == self.input_dim, "SparseMoELayer input shape mismatch."

        expert_out = self._ffn_all_experts(x)                         # [B,T,E,D]
        gates = self.gate_act(torch.einsum("btd,ted->bte", x, self.router))  # [B,T,E]
        out = torch.sum(gates.unsqueeze(-1) * expert_out, dim=2)  # [B,T,D]
        return out
