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

from .auto_split_tokenizer import AutoSplitTokenizer
from .chunk_tokenizer import ChunkTokenizer
from .field_wise_tokenizer import FieldWiseTokenizer
from .random_split_tokenizer import RandomSplitTokenizer


TOKENIZER_TYPES = ("Chunk", "Auto", "Field", "Random")


def build_unified_tokenizer(tokenizer_type,
                            input_dim,
                            field_dim,
                            token_dim,
                            num_tokens,
                            num_fields,
                            activation="SiLU",
                            layer_norm=True):
    """Build a tokenizer used by unified ranking models.

    Returns ``(module, output_token_count, expects_field_input)``. Chunk and
    Auto consume a flattened feature vector. Field and Random consume a
    ``[batch_size, num_fields, field_dim]`` tensor so that field boundaries
    remain explicit.
    """
    normalized_type = str(tokenizer_type).strip().lower()
    canonical_types = {name.lower(): name for name in TOKENIZER_TYPES}

    tokenizer_type = canonical_types[normalized_type]
    if tokenizer_type == "Chunk":
        tokenizer = ChunkTokenizer(
            input_dim=input_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
            activation=activation,
            layer_norm=layer_norm,
        )
        return tokenizer, num_tokens, False

    if tokenizer_type == "Auto":
        tokenizer = AutoSplitTokenizer(
            input_dim=input_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
            activation=activation,
            layer_norm=layer_norm,
        )
        return tokenizer, num_tokens, False

    if tokenizer_type == "Field":
        tokenizer = FieldWiseTokenizer(
            input_dim=field_dim,
            token_dim=token_dim,
            num_tokens=num_fields,
            activation=activation,
            layer_norm=layer_norm,
        )
        return tokenizer, num_fields, True

    tokenizer = RandomSplitTokenizer(
        input_dim=field_dim,
        token_dim=token_dim,
        num_tokens=num_tokens,
        num_fields=num_fields,
        activation=activation,
        layer_norm=layer_norm,
    )
    return tokenizer, num_tokens, True
