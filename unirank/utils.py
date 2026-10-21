# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import logging
import logging.config
import yaml
import glob
import json
import re
import numpy as np
import pyarrow.parquet as pq
from collections import OrderedDict
from pathlib import Path


def resolve_parquet_files(data_path):
    """Resolve a Parquet file, directory, glob, or sequence of paths."""
    if isinstance(data_path, (list, tuple)):
        files = []
        for path in data_path:
            files.extend(resolve_parquet_files(path))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {data_path}")
        return files

    data_path = str(data_path)
    if any(char in data_path for char in ("*", "?", "[")):
        files = sorted(glob.glob(data_path))
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {data_path}")
        return files

    path = Path(data_path)
    if path.is_dir():
        files = sorted(str(file) for file in path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in directory: {data_path}")
        return files

    if path.suffix != ".parquet":
        candidate = Path(f"{data_path}.parquet")
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Parquet path not found: {data_path}")
    return [str(path)]


def get_parquet_schema_names(files):
    """Return column names from the first Parquet file."""
    if isinstance(files, (str, Path)):
        files = resolve_parquet_files(files)
    if not files:
        raise FileNotFoundError("No parquet files found for schema inference.")
    return set(pq.ParquetFile(files[0]).schema.names)


def is_sequence_like(value):
    return isinstance(value, (list, tuple, np.ndarray))


def is_sequence_column(series):
    """Return whether the first non-null value is sequence-like."""
    for value in series:
        if value is not None:
            return is_sequence_like(value)
    return False


def dataframe_to_darray(dataframe):
    """Convert a DataFrame to a dense array and expanded column index."""
    column_index = {}
    arrays = []
    index = 0

    for column in dataframe.columns:
        series = dataframe[column]
        if is_sequence_column(series):
            array = np.array(series.to_list())
            if array.ndim == 1:
                array = series.to_numpy()
                column_index[column] = index
                index += 1
            else:
                sequence_length = array.shape[1]
                column_index[column] = list(range(index, index + sequence_length))
                index += sequence_length
        else:
            array = series.to_numpy()
            column_index[column] = index
            index += 1
        arrays.append(array)

    if not arrays:
        raise ValueError("No columns were loaded from parquet file.")
    return np.column_stack(arrays), column_index


def extract_part_id(file_path):
    """Extract the integer ID from a part-NNNNN.parquet filename."""
    match = re.fullmatch(r"part-(\d+)\.parquet", Path(file_path).name)
    if match is None:
        raise ValueError(f"Invalid blocked parquet filename: {file_path}")
    return int(match.group(1))


def build_part_file_map(path_like):
    """Map blocked Parquet part IDs to file paths."""
    part_files = {}
    for file_path in resolve_parquet_files(path_like):
        part_id = extract_part_id(file_path)
        if part_id in part_files:
            raise ValueError(f"Duplicate part id found: part-{part_id:05d}")
        part_files[part_id] = file_path
    return part_files


def find_meta_data_json(path_like, max_depth=8):
    """Find meta_data.json in the path or one of its parents."""
    path = Path(str(path_like)).resolve()
    directory = path if path.is_dir() else path.parent
    candidates = []

    for _ in range(max_depth):
        candidate = directory / "meta_data.json"
        candidates.append(candidate)
        if candidate.is_file():
            return candidate
        if directory.parent == directory:
            break
        directory = directory.parent

    raise FileNotFoundError(
        f"meta_data.json not found from path: {path_like}\n"
        f"Tried: {[str(candidate) for candidate in candidates]}"
    )


def resolve_side_info_path(split, key, explicit_path=None, config=None):
    """Resolve an explicit or split-specific side-information path."""
    if explicit_path is not None:
        return explicit_path
    config = config or {}
    split_key = f"{split}_{key}" if split is not None else None
    if split_key is not None and config.get(split_key) is not None:
        return config[split_key]
    raise ValueError(
        f"Missing side-info path for key='{key}', split='{split}'. "
        f"Expected either explicit `{key}` or `{split_key}` in the configuration."
    )


def estimate_parquet_block_cost(data_file, seq_len_col="seq_len", sample_rows=4096):
    """Estimate block load from row count and sample its average sequence length."""
    parquet_file = pq.ParquetFile(data_file)
    num_rows = int(parquet_file.metadata.num_rows)
    if seq_len_col not in set(parquet_file.schema.names):
        return {"num_rows": num_rows, "avg_seq_len": None, "cost": float(num_rows)}

    sampled = 0
    sequence_sum = 0.0
    for batch in parquet_file.iter_batches(
        batch_size=min(sample_rows, 1024), columns=[seq_len_col]
    ):
        array = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.float64)
        sequence_sum += float(array.sum())
        sampled += int(len(array))
        if sampled >= sample_rows:
            break

    average_length = sequence_sum / sampled if sampled else 0.0
    return {
        "num_rows": num_rows,
        "avg_seq_len": float(average_length),
        "cost": float(num_rows),
    }


def load_config(config_dir, experiment_id):
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params

def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r', encoding='utf-8') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r", encoding="utf-8") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())


class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value

    def get_metrics(self):
        return list(self.kv_pairs.keys())


def not_in_whitelist(element, whitelist=[]):
    if not whitelist:
        return False
    elif type(whitelist) == list:
        return element not in whitelist
    else:
        return element != whitelist
