# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.

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
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import gc
import logging
import argparse
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from unirank.utils import (
    load_config, set_logger, print_to_json, print_to_list
)
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.pytorch.torch_utils import (
    distributed_barrier,
    init_distributed_env,
    is_main_process,
    parse_gpu_ids,
    seed_everything,
    setup_visible_devices,
)
from unirank.preprocess import FeatureProcessor, build_dataset
import model_zoo


if __name__ == '__main__':
    """
    Single card:
      python run_expid.py

    DDP multi-card (single machine):
      torchrun --standalone --nproc_per_node=4 run_expid.py --gpu 0,1,2,3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='RankMixer_KuaiRand_Video_Action', help='The experiment id to run.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ids, e.g. "0" or "0,1,2,3"; use "-1" for cpu')
    parser.add_argument('--enable_bf16', action=argparse.BooleanOptionalAction, default=None,
                        help='覆盖配置中的 BF16 开关；--no-enable_bf16 关闭混合精度。')
    args = vars(parser.parse_args())

    distributed = False
    try:
        gpu_ids = parse_gpu_ids(args['gpu'])
        setup_visible_devices(gpu_ids)

        distributed, rank, local_rank, world_size, local_world_size = init_distributed_env()

        # Basic legality check + distributed initialization
        if distributed:
            if len(gpu_ids) == 0:
                raise ValueError("CPU cannot be used in DDP mode (--gpu -1)")
            if local_world_size != len(gpu_ids):
                raise ValueError(
                    f"LOCAL_WORLD_SIZE({local_world_size}) is inconsistent with --gpu number({len(gpu_ids)})."
                    f"Please ensure that the number of torchrun --nproc_per_node and --gpu are consistent."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available and DDP+NCCL cannot be used.")

            torch.cuda.set_device(local_rank)

            # Only initialize when not initialized to avoid repeated initialization errors.
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available.")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
            # NCCL is responsible for training tensor communication; large verification result objects are aggregated through CPU/Gloo,
            # Avoid gather_object producing large serialized byte tensor on GPU.
            eval_process_group = dist.new_group(
                backend="gloo",
                timeout=timedelta(minutes=60)
            )
            distributed_barrier(local_rank)
        else:
            eval_process_group = None
            # In non-torchrun mode, multiple GPUs are not allowed (to avoid being mistaken for automatic DDP)
            if len(gpu_ids) > 1:
                raise ValueError(
                    "Multiple GPUs detected, but currently not in torchrun mode. \n"
                    "Please use: torchrun --standalone --nproc_per_node=<Number of GPUs> run_expid.py ... --gpu 0,1,..."
                )
            if len(gpu_ids) == 1 and not torch.cuda.is_available():
                raise RuntimeError("GPU specified but CUDA is not available.")

        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)

        # Equipment parameters:
        # - DDP: local_rank maps to CUDA_VISIBLE_DEVICES internal sequence number
        # - Single card GPU: Fixed use of visible devices 0
        if len(gpu_ids) == 0:
            params['gpu'] = -1
        else:
            params['gpu'] = local_rank if distributed else 0

        params['distributed'] = distributed
        params['rank'] = rank
        params['local_rank'] = local_rank
        params['world_size'] = world_size

        # bf16 switch: command line parameters override fields of the same name in config (if present)
        if args['enable_bf16'] is not None:
            params['enable_bf16'] = args['enable_bf16']

        if is_main_process(rank):
            set_logger(params)
            logging.info("Params: " + print_to_json(params))
        else:
            logging.getLogger().handlers = []
            logging.basicConfig(level=logging.ERROR)

        # Each rank uses a different seed offset
        seed_everything(seed=params['seed'] + rank)

        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        feature_map_json = os.path.join(data_dir, "feature_map.json")

        if distributed:
            if is_main_process(rank):
                feature_encoder = FeatureProcessor(**params)
                params["train_data"], params["valid_data"], params["test_data"] = \
                    build_dataset(feature_encoder, **params)

            obj_list = [[
                params.get("train_data", None),
                params.get("valid_data", None),
                params.get("test_data", None)
            ]] if is_main_process(rank) else [[None, None, None]]

            dist.broadcast_object_list(obj_list, src=0)
            params["train_data"], params["valid_data"], params["test_data"] = obj_list[0]
            distributed_barrier(local_rank)
        else:
            feature_encoder = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(feature_encoder, **params)

        feature_map = FeatureMap(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)
        if is_main_process(rank):
            logging.info("Feature specs: " + print_to_json(feature_map.features))

        model_class = getattr(model_zoo, params['model'])
        model = model_class(feature_map, **params)
        model.model_to_device()
        if distributed:
            model.set_eval_process_group(eval_process_group)

        if distributed:
            ddp_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=params.get("find_unused_parameters", True)
            )
            # Rely on the rank_model.py you modified earlier
            model.set_ddp_model(ddp_model)

        if is_main_process(rank):
            model.count_parameters()

        # --------------------
        # Build data iterators
        # --------------------
        if distributed:
            train_params = dict(params)
            train_params.update({
                "distributed": True,
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size
            })
            # All ranks build train and valid. During verification, each rank performs parallel inference and then all_gather converges.
            train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **train_params).make_iterator()
        else:
            train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()


        if distributed and dist.is_initialized():
            distributed_barrier(local_rank)
        model.fit(train_gen, validation_data=valid_gen, **params)

        del train_gen, valid_gen
        gc.collect()

        if params.get("test_data", None):
            if is_main_process(rank):
                logging.info('******** Test evaluation ********')

            test_params = dict(params)
            if distributed:
                test_params.update({
                    "distributed": True,
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size
                })
            else:
                test_params.update({
                    "distributed": False,
                    "rank": 0,
                    "local_rank": 0,
                    "world_size": 1
                })

            test_gen = RankDataLoader(feature_map, stage='test', **test_params).make_iterator()

            if distributed and dist.is_available() and dist.is_initialized():
                distributed_barrier(local_rank)
            model.evaluate(test_gen)

            del test_gen
            gc.collect()

            if distributed and dist.is_available() and dist.is_initialized():
                distributed_barrier(local_rank)
            # 保留最佳 checkpoint，供后续评估、推理和面试演示使用。

    finally:
        if distributed and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
