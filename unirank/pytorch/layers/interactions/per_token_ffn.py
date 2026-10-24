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
import math

class PerTokenFeedForward(nn.Module):
    """
    论文 Section 3.3: Per-token FFN

    与标准 Transformer 共享一个 FFN 不同，这里每个 token 有独立的 FFN 参数。
    目的: 不同 token 对应不同的语义特征子空间，独立参数可避免高频特征主导。

    公式:
        v_t = f_pffn^{t,2}(Gelu(f_pffn^{t,1}(s_t)))

        其中 f_pffn^{t,i}(x) = x @ W_pffn^{t,i} + b_pffn^{t,i}
        W^{t,1} ∈ R^{D x kD},  W^{t,2} ∈ R^{kD x D}
    """
    def __init__(self,
                 input_dim,
                 num_tokens,
                 ffn_expansion=2,
                 net_dropout=0.0):
        super(PerTokenFeedForward, self).__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.hidden_dim = ffn_expansion * input_dim

        # 为每个 token 分配独立的 W1, b1, W2, b2
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.hidden_dim, self.input_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.input_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_tokens):
            nn.init.xavier_uniform_(self.W1[i])
            nn.init.xavier_uniform_(self.W2[i])
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor):  # x: [B, T, D]
        h = torch.einsum("btd,tdk->btk", x, self.W1) + self.b1
        h = self.act(h)
        h = self.dropout(h)
        y = torch.einsum("btk,tkd->btd", h, self.W2) + self.b2
        return y


class ReLURouter(nn.Module):
    """
    论文 Section 3.4: ReLU Routing

    摒弃 Top-k + Softmax，使用 ReLU 门控。
    不同 token 可以激活不同数量的专家。

    G_{i,j} = ReLU(h(s_i))

    稀疏性由 L1 正则控制:
        L_reg = λ * Σ_i Σ_j G_{i,j}
    """
    def __init__(self, input_dim:int, num_experts: int):
        super(ReLURouter, self).__init__()
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T, D]
        Returns:
            gate_values: [batch, T, num_experts] (非负，ReLU 后)
        """
        return F.relu(self.router(x))

class PerTokenMoEFFN(nn.Module):
    """
    论文 Section 3.4: 将 Per-token FFN 扩展为 Sparse-MoE

    每个 token 有自己的一组专家 (Per-token experts)。
    使用全专家 Softmax 训练路由和 ReLU 推理路由，共享专家参数。

    公式:
        G_train = Softmax(h_train(s_i)), G_infer = ReLU(h_infer(s_i))
        v_i = Σ_{j=1}^{N_e} G_{i,j} * e_{i,j}(s_i)

    DTSI 策略:
        - 训练时: 两条路由共享专家，分别预测并接受任务监督。
        - 稀疏正则只更新 h_infer；h_train 不加稀疏约束。
        - 推理时: 仅使用 h_infer；当前实现仍计算全部专家。
    """
    def __init__(
        self,
        num_tokens,
        input_dim,
        num_experts=4,
        ffn_expansion=4,
        l1_lambda=0.01,
        net_dropout=0.0,
    ):
        super(PerTokenMoEFFN, self).__init__()

        if min(num_tokens, input_dim, num_experts, ffn_expansion) <= 0:
            raise ValueError("MoE dimensions and num_experts must be positive")
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be nonnegative")

        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.hidden_dim = ffn_expansion * input_dim
        self.num_experts = num_experts
        self.l1_lambda = l1_lambda

        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        self.hidden_act = nn.GELU()

        # 每个 token 有 num_experts 个专家
        # Expert weights: 每个 token 每个 expert 有独立的 W1, W2
        # W1: [T, num_experts, D, kD],  W2: [T, num_experts, kD, D]
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.hidden_dim, self.input_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.input_dim))

        # 初始化
        for t in range(num_tokens):
            for e in range(num_experts):
                nn.init.kaiming_uniform_(self.W1[t, e], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W2[t, e], a=math.sqrt(5))

        # 训练路由输出 logits，前向时对全部专家做 Softmax；推理路由保留 ReLU。
        self.router_train = nn.Linear(input_dim, num_experts)
        self.router_infer = ReLURouter(input_dim, num_experts)

    def _expert_forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [batch, T, D]
            gate: [batch, T, num_experts]
        Returns:
            [batch, T, D]
        """
        # 对每个 token t, 每个 expert j:
        #   h = Gelu(x_t @ W1[t,j] + b1[t,j])
        #   e_out = h @ W2[t,j] + b2[t,j]
        #   最终: v_t = Σ_j gate[t,j] * e_out

        # x: [B, T, D] -> expert_out: [B, T, E, D]
        h = torch.einsum("btd,tedk->btek", x, self.W1) + self.b1  # [B, T, E, kD]
        h = self.hidden_act(h)
        h = self.dropout(h)

        out = torch.einsum("btek,tekd->bted", h, self.W2) + self.b2  # [B, T, E, D]
        # 加权求和: gate [batch, T, E] * out [batch, T, E, D]
        out = torch.einsum('bte, bted -> btd', gate, out)  # [batch, T, D]
        return out

    def forward(self, x: torch.Tensor, route="infer"):
        """
        Args:
            x: [batch, T, D]
            route: "train" 或 "infer"，与模块的 train/eval 模式分开。
        Returns:
            output: [batch, T, D]
            reg_loss: 已乘 l1_lambda 的 L1 稀疏正则损失
        """
        if route not in ("train", "infer"):
            raise ValueError("route must be 'train' or 'infer'")
        if x.ndim != 3 or x.shape[1:] != (self.num_tokens, self.input_dim):
            raise ValueError("MoE input must have shape [B, num_tokens, input_dim]")
        # 训练路由不加稀疏正则；eval 模式也不计算正则。
        reg_loss = x.new_zeros((), dtype=torch.float32)
        if route == "train":
            # 在专家维以 FP32 计算 Softmax，不做 Top-k，全部专家参与加权求和。
            gate = F.softmax(self.router_train(x), dim=-1, dtype=torch.float32)
        else:
            # 任务预测使用正常计算图，使 infer 路由、专家和上游都能接受任务梯度。
            gate = self.router_infer(x)
            if self.training:
                # 仅正则分支截断输入梯度；任务分支仍正常反传到专家和上游。
                # 只 detach 输入，不 detach 门控输出，保留正则对 infer 路由参数的梯度。
                reg_gate = self.router_infer(x.detach())
                # 门控形状为 [B,T,E]：专家维求和，样本和 token 维取平均，以 FP32 累加。
                reg_loss = self.l1_lambda * reg_gate.float().sum(dim=-1).mean()

        # 两条路由共用 W1、W2、b1、b2，只改变输入表示和专家门控权重。
        return self._expert_forward(x, gate), reg_loss



class PerTokenSwiGLU(nn.Module):
    def __init__(self,
                 input_dim,
                 num_tokens,
                 ffn_expansion=2,
                 net_dropout=0.0):
        super(PerTokenSwiGLU, self).__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.hidden_dim = ffn_expansion * input_dim
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.W3 = nn.Parameter(torch.empty(self.num_tokens, self.hidden_dim, self.input_dim))
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
    def __init__(self, input_dim, ffn_expansion=4, net_dropout=0.0):
        super(SwiGLU, self).__init__()
        hidden_dim = int(input_dim * ffn_expansion)
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, T, D]
        y = self.act(self.W1(x)) * self.W2(x)
        y = self.dropout(y)
        y = self.W3(y)
        return y


# class SparseMoELayer(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  num_tokens,
#                  ffn_expansion=2,
#                  num_experts=4,
#                  net_dropout=0.0):
#         super(SparseMoELayer, self).__init__()
#         self.input_dim = input_dim          # D
#         self.num_tokens = num_tokens        # T
#         self.hidden_dim = ffn_expansion * input_dim  # kD
#         self.num_experts = num_experts      # E

#         self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

#         self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.input_dim, self.hidden_dim))
#         self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.hidden_dim))
#         self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.hidden_dim, self.input_dim))
#         self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.input_dim))

#         self.hidden_act = nn.GELU()
#         self.gate_act = nn.ReLU()

#         self.router = nn.Parameter(torch.empty(self.num_ns_token, self.num_experts, self.input_dim))
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W1)
#         nn.init.xavier_uniform_(self.W2)
#         nn.init.zeros_(self.b1)
#         nn.init.zeros_(self.b2)
#         nn.init.xavier_uniform_(self.router)

#     def _ffn_all_experts(self, x: torch.Tensor):
#         """
#         x: [B, T, D] -> expert_out: [B, T, E, D]
#         """
#         h = torch.einsum("btd,tedk->btek", x, self.W1) + self.b1  # [B, T, E, kD]
#         h = self.hidden_act(h)
#         h = self.dropout(h)
#         expert_out = torch.einsum("btek,tekd->bted", h, self.W2) + self.b2  # [B, T, E, D]
#         return expert_out

#     def forward(self, x: torch.Tensor):
#         B, T, D = x.shape
#         assert T == self.num_ns_token and D == self.input_dim, "SparseMoELayer input shape mismatch."

#         expert_out = self._ffn_all_experts(x)                         # [B,T,E,D]
#         gates = self.gate_act(torch.einsum("btd,ted->bte", x, self.router))  # [B,T,E]
#         out = torch.sum(gates.unsqueeze(-1) * expert_out, dim=2)  # [B,T,D]
#         return out
