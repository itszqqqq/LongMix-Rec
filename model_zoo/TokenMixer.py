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
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention, PerTokenSwiGLU, MultiHeadTokenMixing
from unirank.pytorch.layers.tokenization import ChunkTokenizer, AutoSplitTokenizer


class TokenMixer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="TokenMixer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 attention_dropout=0,
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 attention_dim=None,
                 num_group_token=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 attention_activation_type="SoftMax",
                 **kwargs):
        super(TokenMixer, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_group_token = num_group_token
        self.accumulation_steps = accumulation_steps

        # Track item and non-item feature dimensions
        self.item_info_dim = 0
        self.non_item_dim = 0
        num_field = feature_map.get_num_fields()

        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                num_field = num_field + 1
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.unified_tokenizer_layer = TokenMixerTokenizer(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            num_group_token=num_group_token,
            num_field=num_field
        )

        self.attention_layers = MultiHeadTargetAttention(
            input_dim=self.item_info_dim,
            attention_dim=token_dim if attention_dim is None else attention_dim,
            dropout_rate=attention_dropout,
            attention_activation_type=attention_activation_type,
        )
        self.unified_interaction_layers = TokenMixerBlocks(input_dim=token_dim,
                                         num_tokens=1 + num_group_token,
                                         num_layers=num_layers,
                                         expand=expansion_factor,
                                         net_dropout=net_dropout)

        self.tower = nn.ModuleList([MLP_Block(input_dim=self.unified_interaction_layers.output_dim,
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
        # item_dict contains [history_items..., target_item]
        # Reshape flattened embeddings to B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        seq_pooling_emb = self.attention_layers(target_emb, sequence_emb, mask) # B x item_info_dim

        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=False)       # B x F × embedding_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb.view(batch_size, -1, self.embedding_dim),
                                        seq_pooling_emb.view(batch_size, -1, self.embedding_dim)], dim=1)
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)  # B x (1 + num_group_token) x token_dim

        # unified model
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            unified_tokens
        )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class TokenMixerTokenizer(nn.Module):
    def __init__(self, embedding_dim, token_dim, num_group_token, num_field):
        super(TokenMixerTokenizer, self).__init__()
        input_dim = num_field * embedding_dim
        self.global_token_mlps = AutoSplitTokenizer(input_dim, token_dim, 1)
        self.group_token_mlps = ChunkTokenizer(input_dim, token_dim, num_group_token)

    def forward(self, feature_embeddings):
        global_tokens = self.global_token_mlps(feature_embeddings)
        group_tokens = self.group_token_mlps(feature_embeddings)
        return torch.cat([global_tokens, group_tokens], dim=1)


class TokenMixerBlocks(nn.Module):
    def __init__(self,
                 input_dim,
                 num_tokens,
                 num_layers,
                 expand=2,
                 net_dropout=0.0):
        super(TokenMixerBlocks, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.block_dim = ((input_dim + num_tokens - 1) // num_tokens) * num_tokens
        self.pad_dim = self.block_dim - input_dim
        self.output_dim = self.block_dim

        self.mixer_norms = nn.ModuleList([
            nn.LayerNorm(self.block_dim)
            for _ in range(num_layers)
        ])
        self.pffn_norms = nn.ModuleList([
            nn.LayerNorm(self.block_dim)
            for _ in range(num_layers)
        ])

        self.mixer_layers = nn.ModuleList([
            MultiHeadTokenMixing(input_dim=self.block_dim, num_tokens=num_tokens)
            for _ in range(num_layers)
        ])
        self.revert_layers = nn.ModuleList([
            MultiHeadTokenMixing(input_dim=self.block_dim, num_tokens=num_tokens)
            for _ in range(num_layers)
        ])
        self.pffn_layers1 = nn.ModuleList([
            PerTokenSwiGLU(input_dim=self.block_dim,
                           num_tokens=num_tokens,
                           ffn_expansion=expand,
                           net_dropout=net_dropout)
            for _ in range(num_layers)
        ])
        self.pffn_layers2 = nn.ModuleList([
            PerTokenSwiGLU(input_dim=self.block_dim,
                                num_tokens=num_tokens,
                                ffn_expansion=expand,
                                net_dropout=net_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor): # B x T x token_dim
        if self.pad_dim > 0:
            x = F.pad(x, (0, self.pad_dim))
        for i in range(self.num_layers):
            mix_x = self.mixer_layers[i](self.mixer_norms[i](x))
            mix_x = self.pffn_layers1[i](mix_x)
            x = self.revert_layers[i](mix_x) + x
            x = self.pffn_layers2[i](self.pffn_norms[i](x)) + x
        return x
