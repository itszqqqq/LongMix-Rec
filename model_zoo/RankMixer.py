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
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention, PerTokenFeedForward, MultiHeadTokenMixing
from unirank.pytorch.layers.interactions.per_token_ffn import PerTokenMoEFFN
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import build_unified_tokenizer


class RankMixer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="RankMixer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",   
                 tower_hidden_units=[128, 64],  # 每个任务预测塔的隐藏层
                 attention_dropout=0,
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=6,
                 token_dim=256,
                 attention_dim=None,   # 历史 attention 内部 Q/K/V 的宽度
                 num_ns_token=4,
                 tokenizer_type="Chunk",
                 net_dropout=0,
                 accumulation_steps=1,   # 训练时梯度累积次数
                 attention_activation_type="SoftMax",
                 **kwargs):
        super(RankMixer, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps   

        # Track item and non-item feature dimensions
        self.item_info_dim = 0  
        self.non_item_dim = 0
        self.num_item_fields = 0  # item的字段数量
        self.num_non_item_fields = 0  # non-item的字段数量
        
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim  
                self.num_item_fields += 1    
            else:
                self.non_item_dim += emb_dim   
                self.num_non_item_fields += 1

        # Initialize embedding layer and tokenizer 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)  
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim    # sum_emb_out_dim() 统计字段输出向量拼接后的总维度；历史摘要，维度正好是 self.item_info_dim
        self.num_tokenizer_fields = self.num_non_item_fields + 2 * self.num_item_fields

        (self.unified_tokenizer_layer,  # tokenizer 模块
         self.num_ns_token,   # 实际输出的 token 数
         self.tokenizer_uses_field_input) = build_unified_tokenizer(
            tokenizer_type=tokenizer_type,  # tokenizer 类型；Chunk: 将完整向量连续切成若干块，每块独立投影
            input_dim=input_dim,      # 输入拼接向量的总维度
            field_dim=embedding_dim,    # 每个字段的 embedding 维度
            token_dim=token_dim,    # tokenizer 输出 token 的维度
            num_tokens=num_ns_token,   # tokenizer 输出 token 的数量
            num_fields=self.num_tokenizer_fields,   # tokenizer 输入的字段数量
        )
        self.tokenizer_type = str(tokenizer_type).strip().title()

        # 历史序列的 attention 层
        self.attention_layers = MultiHeadTargetAttention(
            input_dim=self.item_info_dim,    # 输入 item 向量的维度
            attention_dim=token_dim if attention_dim is None else attention_dim,  # 历史 attention 内部 Q/K/V 的维度。attention_dim=None 时，代码借用 token_dim 作为内部宽度；两者数值相同，不代表它们具有同一个作用
            dropout_rate=attention_dropout,
            attention_activation_type=attention_activation_type,
        )

        self.unified_interaction_layers = RankMixerBlock(input_dim=token_dim,
                                         num_ns_token=self.num_ns_token,
                                         num_layers=num_layers,
                                         ffn_expansion=expansion_factor,
                                         net_dropout=net_dropout)

        self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim,
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=tower_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout)
                                    for _ in range(num_tasks)])
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]
        # item_dict 包含 [history_items..., target_item]
        # Reshape flattened embeddings to B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)    # （B, T+1, item_info_dim），其中 T 是历史 item 的数量，item_info_dim 是每个 item 的 embedding 维度

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        seq_pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)   # B x embedding_dim  与当前候选相关的历史摘要

        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb, seq_pooling_emb], dim=-1)
        if self.tokenizer_uses_field_input:  # 如果 tokenizer 需要明确的字段边界
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_tokenizer_fields, self.embedding_dim
            )
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)           # B x num_tokens x token_dim 

        # unified model
        unified_tokens = self.activation_checkpoint(    # 执行多层 RankMixer 特征交互
            self.unified_interaction_layers,
            unified_tokens
        )
        # unified_tokens = self.unified_interaction_layers(unified_tokens)
        # activation_checkpoint 控制激活重计算。训练时启用后，反向传播会重算部分前向，减少需要保存的激活，以额外计算换显存

        bottom_output = unified_tokens.mean(dim=1)  # 平均特征 token，得到共享表示 [B,4,256] → [B,256]
        
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]  # num_tasks 个 [B,1] 张量
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]    # 对二分类任务，output_activation[i] 是 Sigmoid； y_pred[i]：预测概率[B,1]
        return_dict = {}
        labels = self.feature_map.labels

        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class RankMixerBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_ns_token,
                 num_layers,
                 ffn_expansion=2,
                 net_dropout=0.0,
                 num_experts=0,  # 0 表示普通 Per-token FFN；正数表示每个 token 的专家数量。
                 l1_lambda=0.01):  # 推理路由的 L1 稀疏正则系数。
        super(RankMixerBlock, self).__init__()
        if num_layers < 0 or (num_experts > 0 and num_layers == 0):
            raise ValueError("MoE requires num_layers > 0; dense num_layers must be nonnegative")
        if num_experts < 0:
            raise ValueError("num_experts must be nonnegative")
        self.num_layers = num_layers   
        self.use_moe = num_experts > 0

        self.mixer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        self.pffn_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])

        # Multi-head Token Mixing
        self.mixer_layers = nn.ModuleList([
            MultiHeadTokenMixing(input_dim=input_dim, num_tokens=num_ns_token)
            for _ in range(num_layers)
        ])

        # 每层只创建一组专家参数，训练路由和推理路由共用这一组专家。
        self.pffn_layers = nn.ModuleList([
            PerTokenFeedForward(input_dim=input_dim,
                           num_tokens=num_ns_token,
                           ffn_expansion=ffn_expansion,
                           net_dropout=net_dropout)
            if not self.use_moe else
            PerTokenMoEFFN(input_dim=input_dim,
                          num_tokens=num_ns_token,
                          num_experts=num_experts,
                          ffn_expansion=ffn_expansion,
                          l1_lambda=l1_lambda,
                          net_dropout=net_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, route="infer"):
        # 每次调用独立计算一条完整路径；route 指定各层使用哪个路由器。
        reg_loss = x.new_zeros((), dtype=torch.float32)
        for i in range(self.num_layers):
            x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
            if self.use_moe:
                ffn_output, layer_reg = self.pffn_layers[i](self.pffn_norms[i](x), route)
                x = ffn_output + x
                # layer_reg 已乘正则系数；训练路由和 eval 模式均返回零正则。
                reg_loss = reg_loss + layer_reg
            else:
                x = self.pffn_layers[i](self.pffn_norms[i](x)) + x

        if self.use_moe:
            # 对各层正则取平均，避免正则尺度随层数增加。
            return x, reg_loss / self.num_layers
        return x

