# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2026. The UniRank Library. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import torch.nn as nn
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as torch_activation_checkpoint
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import os, sys
import logging
from unirank.metrics import evaluate_metrics
from unirank.pytorch.torch_utils import get_device, get_optimizer, get_loss
from unirank.utils import Monitor, not_in_whitelist
from tqdm import tqdm
from contextlib import nullcontext

try:
    from torch.distributed.algorithms.join import Join
except Exception:
    Join = None

class Float32Sigmoid(nn.Module):
    """Keep probability precision before BCE, including under BF16 autocast."""

    def forward(self, logits):
        # Casting probabilities after a BF16 sigmoid cannot undo rounding to 1.
        return torch.sigmoid(logits.float())


class BaseModel(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="BaseModel",
                 task="binary_classification",
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps  # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]

        # DDP related
        self.distributed = kwargs.get("distributed", False)
        self.rank = kwargs.get("rank", 0)
        self.local_rank = kwargs.get("local_rank", 0)
        self.world_size = kwargs.get("world_size", 1)
        self.force_distributed_eval = kwargs.get("force_distributed_eval", False)
        self._ddp_model = None
        self._eval_process_group = None

        # bf16 mixed precision switch
        self.enable_bf16 = kwargs.get("enable_bf16", True)
        self.enable_torch_compile = kwargs.get("enable_torch_compile", True)
        self.gradient_checkpointing = bool(kwargs.get("gradient_checkpointing", False))
        self._torch_compile_enabled = False
        self.dense_optimizer_name = kwargs.get("dense_optimizer", "AdamW")
        self.sparse_optimizer_name = kwargs.get("sparse_optimizer", "Adagrad")
        self.dense_learning_rate = kwargs.get("dense_learning_rate", 1e-4)
        self.sparse_learning_rate = kwargs.get("sparse_learning_rate", 0.05)
        self.dense_weight_decay = kwargs.get("dense_weight_decay", 0)
        self.dense_optimizer = None
        self.sparse_optimizer = None
        self.optimizers = []
        self.optimizer = None
        self._dense_params = []
        self._sparse_params = []

    def activation_checkpoint(self, function, *args):
        """Run a module/function with non-reentrant activation checkpointing.

        Non-reentrant checkpointing is compatible with DDP, including the current
        find_unused_parameters=True setup. During evaluation or when the feature is
        disabled, this is a zero-behavior-change direct call.
        """
        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return function(*args)
        return torch_activation_checkpoint(function, *args, use_reentrant=False)

    def compile(self, optimizer, loss, lr):
        self._compile_optimizers(optimizer, lr)
        self.loss_fn = get_loss(loss)
        self._maybe_enable_torch_compile()

    def _split_dense_sparse_parameters(self):
        sparse_param_ids = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        sparse_param_ids.add(id(param))

        dense_params, sparse_params = [], []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in sparse_param_ids:
                sparse_params.append(param)
            else:
                dense_params.append(param)
        return dense_params, sparse_params

    def _compile_optimizers(self, dense_optimizer=None, dense_lr=None):
        dense_optimizer = dense_optimizer or self.dense_optimizer_name
        dense_lr = self.dense_learning_rate if dense_lr is None else dense_lr
        self._dense_params, self._sparse_params = self._split_dense_sparse_parameters()
        self.dense_optimizer = get_optimizer(
            dense_optimizer,
            self._dense_params,
            dense_lr,
            weight_decay=self.dense_weight_decay
        )
        self.sparse_optimizer = get_optimizer(
            self.sparse_optimizer_name,
            self._sparse_params,
            self.sparse_learning_rate,
            weight_decay=0
        )
        self.optimizers = [opt for opt in [self.dense_optimizer, self.sparse_optimizer] if opt is not None]
        if not self.optimizers:
            raise RuntimeError("No trainable parameters found for optimizer.")
        self.optimizer = self.dense_optimizer if self.dense_optimizer is not None else self.optimizers[0]

        if self._is_main_process():
            logging.info(
                "Optimizers: dense=%s(lr=%s, weight_decay=%s, params=%d), sparse=%s(lr=%s, params=%d)",
                dense_optimizer,
                dense_lr,
                self.dense_weight_decay,
                sum(p.numel() for p in self._dense_params),
                self.sparse_optimizer_name if self.sparse_optimizer is not None else "None",
                self.sparse_learning_rate,
                sum(p.numel() for p in self._sparse_params),
            )

    def _iter_optimizers(self):
        return self.optimizers

    def _optimizer_zero_grad(self):
        for optimizer in self._iter_optimizers():
            optimizer.zero_grad()

    def _optimizer_step(self):
        for optimizer in self._iter_optimizers():
            optimizer.step()

    def _clip_dense_gradients(self):
        dense_params_with_grad = [p for p in self._dense_params if p.grad is not None]
        if dense_params_with_grad:
            nn.utils.clip_grad_norm_(dense_params_with_grad, self._max_gradient_norm)

    def _format_learning_rates(self):
        dense_lr = None
        sparse_lr = None
        if self.dense_optimizer is not None and self.dense_optimizer.param_groups:
            dense_lr = self.dense_optimizer.param_groups[0]["lr"]
        if self.sparse_optimizer is not None and self.sparse_optimizer.param_groups:
            sparse_lr = self.sparse_optimizer.param_groups[0]["lr"]
        if sparse_lr is None:
            return f"{dense_lr:.2e}"
        if dense_lr is None:
            return f"sparse:{sparse_lr:.2e}"
        return f"dense:{dense_lr:.2e}/sparse:{sparse_lr:.2e}"

    def _maybe_enable_torch_compile(self):
        if self._torch_compile_enabled or (not self.enable_torch_compile):
            return
        if not hasattr(torch, "compile"):
            logging.warning("torch.compile is not available in this PyTorch build, skip compile().")
            return
        compile_targets = []
        container_targets = []
        for name, module in self.named_children():
            if name in {"output_activation"}:
                continue
            has_trainable_params = any(p.requires_grad for p in module.parameters())
            has_sparse_embedding = any(isinstance(m, nn.Embedding) for m in module.modules())
            if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                iterator = module.items() if isinstance(module, nn.ModuleDict) else enumerate(module)
                for key, submodule in iterator:
                    sub_has_trainable_params = any(p.requires_grad for p in submodule.parameters())
                    sub_has_sparse_embedding = any(isinstance(m, nn.Embedding) for m in submodule.modules())
                    if sub_has_trainable_params and not sub_has_sparse_embedding:
                        container_targets.append((name, module, key, submodule))
                continue
            if has_trainable_params and not has_sparse_embedding:
                compile_targets.append((name, module))
        if not compile_targets and not container_targets:
            logging.warning("No dense child modules found on model, skip compile().")
            return
        try:
            logging.info("************ compile start ************")
            for name, module in compile_targets:
                setattr(
                    self,
                    name,
                    torch.compile(
                        module,
                        backend="inductor"
                    )
                )
                logging.info("Compiled dense module: %s", name)
            for name, container, key, submodule in container_targets:
                container[key] = torch.compile(
                    submodule,
                    backend="inductor"
                )
                logging.info("Compiled dense module: %s.%s", name, key)
            self._torch_compile_enabled = True
        except Exception as ex:
            logging.warning("torch.compile failed and will be skipped: %s", ex)

    def set_ddp_model(self, ddp_model):
        """Attach DDP wrapper for forward/backward only, without module registration cycle."""
        if "_ddp_model" in self._modules:
            self._modules.pop("_ddp_model")
        object.__setattr__(self, "_ddp_model", ddp_model)

    def set_eval_process_group(self, process_group):
        """Attach a CPU process group used for large evaluation-result objects."""
        object.__setattr__(self, "_eval_process_group", process_group)

    def _is_distributed(self):
        return self.distributed and dist.is_available() and dist.is_initialized()

    def _is_main_process(self):
        return (not self._is_distributed()) or (self.rank == 0)

    def _distributed_barrier(self):
        """Synchronize ranks while making the NCCL device mapping explicit."""
        if not self._is_distributed():
            return
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def _train_forward_model(self):
        return self._ddp_model if self._ddp_model is not None else self

    def _get_amp_context(self):
        """
        Returns the bf16 autocast context manager.
        - enable_bf16=True and device is CUDA: use torch.autocast(cuda, bfloat16)
        - enable_bf16=True and the device is CPU: use torch.autocast(cpu, bfloat16)
        - enable_bf16=False: returns nullcontext() without any precision conversion
        """
        if self.enable_bf16:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _sync_stop_flag_from_main(self):
        """Broadcast early-stop flag from rank0 to all ranks."""
        if not self._is_distributed():
            return
        if self.device.type == "cuda":
            flag = torch.tensor([1 if self._stop_training else 0], device=self.device, dtype=torch.int32)
        else:
            flag = torch.tensor([1 if self._stop_training else 0], dtype=torch.int32)
        dist.broadcast(flag, src=0)
        self._stop_training = bool(flag.item())

    def _sync_lr_from_main(self):
        """Broadcast learning rate from rank0 to all ranks."""
        if not self._is_distributed():
            return
        param_groups = [pg for optimizer in self._iter_optimizers() for pg in optimizer.param_groups]
        if not param_groups:
            return
        lr_tensor = torch.tensor([pg["lr"] for pg in param_groups], device=self.device, dtype=torch.float64)
        dist.broadcast(lr_tensor, src=0)
        for param_group, new_lr in zip(param_groups, lr_tensor.tolist()):
            param_group["lr"] = new_lr

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def compute_loss(self, return_dict, y_true):
        return self.add_loss(return_dict, y_true)

    def reset_parameters(self):
        def default_reset_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        def custom_reset_params(m):
            # initialize layers with customized init_weights()
            if hasattr(m, 'init_weights'):
                m.init_weights()

        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def get_inputs(self, inputs, feature_source=None, return_multi_masks=False):
        """
        Args:
            inputs: (batch_dict, item_dict, mask) or (batch_dict, item_dict, mask, multi_masks)
            feature_source: optional feature source filtering
            return_multi_masks: bool, default False
                - False: Return (X_dict, item_dict, mask)
                - True : Returns (X_dict, item_dict, mask, multi_masks)
        """
        if len(inputs) == 4:
            batch_dict, item_dict, mask, multi_masks = inputs
        elif len(inputs) == 3:
            batch_dict, item_dict, mask = inputs
            multi_masks = None
        else:
            raise ValueError(f"Unexpected inputs length: {len(inputs)}. Expected 3 or 4.")

        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)

        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)

        mask = mask.to(self.device)

        if not return_multi_masks:
            return X_dict, item_dict, mask

        if multi_masks is None:
            multi_masks = [mask for _ in range(self.num_tasks)]
        else:
            if isinstance(multi_masks, (list, tuple)):
                multi_masks = [m.to(self.device) for m in multi_masks]
            elif torch.is_tensor(multi_masks):
                multi_masks = multi_masks.to(self.device)
                if multi_masks.dim() == 3 and multi_masks.size(0) == self.num_tasks:
                    multi_masks = [multi_masks[i] for i in range(self.num_tasks)]
                else:
                    multi_masks = [multi_masks for _ in range(self.num_tasks)]
            else:
                multi_masks = [mask for _ in range(self.num_tasks)]
        return X_dict, item_dict, mask, multi_masks

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)
        self._optimizer_to_device()

    def _optimizer_to_device(self):
        if not getattr(self, "optimizers", None):
            return
        for optimizer in self._iter_optimizers():
            for param, state in optimizer.state.items():
                param_device = param.device
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device=param_device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        dense_lr = None
        for optimizer in self._iter_optimizers():
            for param_group in optimizer.param_groups:
                reduced_lr = max(param_group["lr"] * factor, min_lr)
                param_group["lr"] = reduced_lr
                if optimizer is self.dense_optimizer and dense_lr is None:
                    dense_lr = reduced_lr
        if dense_lr is not None:
            return dense_lr
        return self._iter_optimizers()[0].param_groups[0]["lr"]

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        """
        DDP + blocked dataloader training instructions:
        - When blocked=True, the local batch numbers of different ranks may not be completely consistent;
        - During the DDP training phase, if some ranks enter eval/barrier first, while other ranks are still in backward allreduce,
          This will cause the NCCL collective order to be inconsistent and eventually timeout;
        - Therefore, in DDP + blocked mode, eval within step is forcibly prohibited, and eval will only be unified after all ranks complete the current epoch;
        - At the same time, use torch.distributed.algorithms.join.Join to process uneven inputs, so that the rank that ends first
          Shadow subsequent collectives to prevent the last few training batches from getting stuck.
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0

        # Whether the current dataloader is in blocked mode.
        # UniRankDataloader will set self.blocked in __init__.
        self._blocked_training = bool(getattr(data_generator, "blocked", False))

        # DDP + blocked forces epoch-end eval only to avoid each rank due to local len(data_generator)
        # Enter eval at different steps, causing the collective order to be disordered.
        self._epoch_end_eval_only = bool(self._is_distributed() and self._blocked_training)

        if self._eval_steps is None:
            # Original semantics: None means verify once every epoch.
            # In non-blocked scenarios, the logic of train_epoch internal triggering by _steps_per_epoch is still retained;
            # In the DDP + blocked scenario, eval will be skipped internally in train_epoch, and fit() will unify eval at the end of epoch.
            self._eval_steps = self._steps_per_epoch

        if self._is_main_process():
            logging.info("BF16 mixed precision: {}".format(self.enable_bf16))
            logging.info("Gradient checkpointing: {}".format(self.gradient_checkpointing))
            logging.info("Start training: {} local batches/epoch".format(self._steps_per_epoch))
            logging.info("DDP blocked training: {}".format(self._epoch_end_eval_only))
            if self._epoch_end_eval_only:
                logging.info("Disable step-wise eval and use epoch-end synchronized eval for DDP + blocked dataloader.")
            logging.info("************ Epoch=1 start ************")

        for epoch in range(epochs):
            self._epoch_index = epoch

            # DDP: make DistributedSampler / blocked iterable shuffle differently every epoch
            if hasattr(data_generator, "set_epoch"):
                data_generator.set_epoch(epoch)
            elif hasattr(data_generator, "sampler") and hasattr(data_generator.sampler, "set_epoch"):
                data_generator.sampler.set_epoch(epoch)

            use_ddp_join = bool(
                self._is_distributed()
                and self._ddp_model is not None
                and Join is not None
                and self._blocked_training
            )

            if use_ddp_join:
                # Uneven inputs scenario: When some ranks run out of data first, Join will let them continue to shadow
                # Other ranks follow the collective in backward to avoid the last few allreduces getting stuck.
                with Join([self._ddp_model], throw_on_early_termination=False):
                    epoch_loss, epoch_batches = self.train_epoch(data_generator)
            else:
                if self._is_distributed() and self._blocked_training and Join is None:
                    logging.warning(
                        "torch.distributed.algorithms.join.Join is unavailable. "
                        "DDP + blocked with uneven local batches may still hang."
                    )
                epoch_loss, epoch_batches = self.train_epoch(data_generator)

            # DDP + blocked: Only allow all ranks to unify eval after completing training epoch.
            if self._epoch_end_eval_only and (not self._stop_training):
                if self._is_main_process():
                    denom = max(1, epoch_batches)
                    logging.info("Train loss: {:.6f}".format(epoch_loss / denom))

                if self._is_distributed():
                    self._distributed_barrier()

                self.eval_step()

                if self._is_distributed():
                    self._distributed_barrier()
                    self._sync_stop_flag_from_main()
                    self._sync_lr_from_main()

            if self._stop_training:
                break
            else:
                if self._is_main_process():
                    logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
                    if epoch + 1 < epochs:
                        logging.info("************ Epoch={} start ************".format(self._epoch_index + 2))

        if self._is_main_process():
            logging.info("Training finished.")
            logging.info("Load best model: {}".format(self.checkpoint))

        # Ensure checkpoint is fully written before all ranks load
        if self._is_distributed():
            self._distributed_barrier()
        self.load_weights(self.checkpoint)
        if self._is_distributed():
            self._distributed_barrier()

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        # Only rank0 decides early-stop and saves checkpoints
        if not self._is_main_process():
            return

        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
                (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}" \
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        # All ranks participate in verification reasoning, and the index is calculated by rank 0 after aggregation through all_gather
        if self._is_main_process():
            logging.info('Evaluation @epoch {} - batch {}: '.format(
                self._epoch_index + 1, self._batch_index + 1))

        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())

        if self._is_main_process():
            self.checkpoint_and_earlystop(val_logs)

        self.train()  # All ranks need to restore train mode

    def train_step(self, batch_data):
        # 最后一组不足 accumulation_steps 时，也须在验证/保存前完成更新。
        is_last_batch = self._batch_index + 1 == self._steps_per_epoch
        is_update_step = ((self._batch_index + 1) % self.accumulation_steps == 0) or is_last_batch
        window_start = (self._batch_index // self.accumulation_steps) * self.accumulation_steps
        accumulation_size = min(self.accumulation_steps, self._steps_per_epoch - window_start)
        use_no_sync = (
                self._is_distributed()
                and (self._ddp_model is not None)
                and self.accumulation_steps > 1
                and (not is_update_step)
        )
        sync_ctx = self._ddp_model.no_sync() if use_no_sync else nullcontext()
        amp_ctx = self._get_amp_context()

        with sync_ctx:
            # Forward propagation only within amp_ctx, enjoy bf16 acceleration
            with amp_ctx:
                return_dict = self._train_forward_model()(batch_data)

            # After exiting autocast, convert all floating point predictions back to float32,
            # Avoid BCELoss / binary_cross_entropy reporting errors under bf16
            return_dict = {
                k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                for k, v in return_dict.items()
            }

            y_true = self.get_labels(batch_data)
            loss = self.compute_loss(return_dict, y_true)
            (loss / accumulation_size).backward()

        if is_update_step:
            self._clip_dense_gradients()
            self._optimizer_step()
            self._optimizer_zero_grad()

        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0.0
        num_batches = 0
        self.train()
        self._optimizer_zero_grad()

        # Each rank can display tqdm
        use_tqdm = (self._verbose > 0)
        if use_tqdm:
            batch_iterator = tqdm(
                data_generator,
                disable=False,
                file=sys.stdout,
                desc=f"Rank {self.rank} | Epoch {self._epoch_index + 1}",
                position=self.rank,
                leave=(self.rank == 0),
                dynamic_ncols=True
            )
        else:
            batch_iterator = data_generator

        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            num_batches += 1

            loss = self.train_step(batch_data)
            loss_value = loss.item()
            train_loss += loss_value

            if use_tqdm:
                avg_loss = train_loss / max(1, num_batches)
                batch_iterator.set_postfix(
                    loss=f"{loss_value:.6f}",
                    avg_loss=f"{avg_loss:.6f}",
                    lr=self._format_learning_rates()
                )

            # In the DDP + blocked scenario, eval within the step of the training phase is prohibited.
            # Reason: The local batch number of each rank of blocked dataloader may be different, and a certain rank enters eval first.
            # And the other rank is still in backward allreduce, which will cause the NCCL collective order to be inconsistent.
            do_step_eval = (
                (not getattr(self, "_epoch_end_eval_only", False))
                and self._eval_steps is not None
                and self._eval_steps > 0
                and self._total_steps % self._eval_steps == 0
            )

            if do_step_eval:
                if self._is_main_process():
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0.0

                self.eval_step()

                if self._is_distributed():
                    self._distributed_barrier()
                    self._sync_stop_flag_from_main()
                    self._sync_lr_from_main()

            if self._stop_training:
                break

        # 提前停止发生在累积窗口中间时，提交尚未更新的梯度。
        if (num_batches > 0
                and self._batch_index + 1 != self._steps_per_epoch
                and (self._batch_index + 1) % self.accumulation_steps != 0):
            self._clip_dense_gradients()
            self._optimizer_step()
            self._optimizer_zero_grad()

        return train_loss, num_batches

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            raw_generator = data_generator
            y_pred = []
            y_true = []
            group_id = []

            # Change: Each rank displays its own verification progress
            if self._verbose > 0:
                data_generator = tqdm(
                    data_generator,
                    disable=False,
                    file=sys.stdout,
                    desc=f"Rank {self.rank} | Eval",
                    position=self.rank,
                    leave=(self.rank == 0),
                    dynamic_ncols=True
                )

            amp_ctx = self._get_amp_context()
            with amp_ctx:
                for batch_data in data_generator:
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].float().data.cpu().numpy().reshape(-1))
                    y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                    if self.feature_map.group_id is not None:
                        group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))

            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None

            # ---- Distributed consistency check: All ranks must agree on "whether to perform distributed evaluation aggregation" ----
            _distributed_eval = False
            _is_sampler_distributed_eval = False
            _is_blocked_distributed_eval = False

            if self._is_distributed():
                _is_sampler_distributed_eval = bool(
                    hasattr(raw_generator, "sampler")
                    and isinstance(raw_generator.sampler, DistributedSampler)
                )

                _is_blocked_distributed_eval = bool(
                    getattr(raw_generator, "blocked", False)
                    and hasattr(raw_generator, "dataset")
                    and getattr(raw_generator.dataset, "distributed", False)
                )

                local_flag = int(_is_sampler_distributed_eval or _is_blocked_distributed_eval)

                flag_t = torch.tensor([local_flag], dtype=torch.int32, device=self.device)
                gathered_flags = [torch.zeros_like(flag_t) for _ in range(self.world_size)]
                dist.all_gather(gathered_flags, flag_t)
                flags = [int(x.item()) for x in gathered_flags]

                if len(set(flags)) != 1:
                    raise RuntimeError(
                        f"[Rank {self.rank}] Inconsistent distributed-eval flags across ranks: {flags}. "
                        "This can cause a collective deadlock. Please ensure that the validation DataLoader configuration of each rank is consistent."
                    )

                _distributed_eval = bool(flags[0])

                if self.force_distributed_eval and not _distributed_eval:
                    raise RuntimeError(
                        "force_distributed_eval=True, but the currently evaluated DataLoader does not have distributed sharding enabled."
                    )

            # ---- Distributed evaluation aggregation ----
            if _distributed_eval:
                if _is_blocked_distributed_eval:
                    local_samples_t = torch.tensor([len(y_true)], dtype=torch.int64, device=self.device)
                    dist.all_reduce(local_samples_t, op=dist.ReduceOp.SUM)
                    total_samples = int(local_samples_t.item())
                else:
                    total_samples = len(raw_generator.dataset)

                y_pred, y_true, group_id = self._gather_eval_results(
                    y_pred, y_true, group_id, total_samples
                )

                # Non-main processes are not counted as indicators and an empty dictionary is returned directly.
                if not self._is_main_process():
                    return OrderedDict()

            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)

            if self._is_main_process():
                logging.info('[Metrics] ' + ' - '.join(
                    '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))

            return val_logs

    def _gather_eval_results(self, y_pred, y_true, group_id, total_samples):
        """
        More robust distributed evaluation aggregation:
        - Use a separate Gloo process group to aggregate objects on the CPU to avoid NCCL picking large
          Converting objects to GPU byte tensor causes memory spikes or NCCL unhandled CUDA errors
        - Only summarized to rank0, reducing the memory pressure of other ranks
        - Rank0 is cropped to total_samples after splicing (removing DistributedSampler padding)
        """
        payload = {
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "group_id": (None if group_id is None else np.asarray(group_id))
        }

        process_group = self._eval_process_group
        if process_group is None and dist.get_backend() == "nccl":
            raise RuntimeError(
                "Distributed evaluation over NCCL requires a CPU/Gloo process group. "
                "Call model.set_eval_process_group(...) before evaluate()."
            )

        group_world_size = dist.get_world_size(group=process_group)

        # Only rank0 collection reduces memory and communication pressure. Explicitly pass in the Gloo group to ensure objects don't fall to the GPU.
        if hasattr(dist, "gather_object"):
            if self._is_main_process():
                gathered = [None for _ in range(group_world_size)]
                dist.gather_object(payload, gathered, dst=0, group=process_group)
            else:
                dist.gather_object(payload, None, dst=0, group=process_group)
                return None, None, None
        else:
            # Bottom line: All ranks are collected (old version of torch)
            gathered = [None for _ in range(group_world_size)]
            dist.all_gather_object(gathered, payload, group=process_group)
            if not self._is_main_process():
                return None, None, None

        y_pred = np.concatenate([g["y_pred"] for g in gathered], axis=0)[:total_samples]
        y_true = np.concatenate([g["y_true"] for g in gathered], axis=0)[:total_samples]

        has_gid = any(g["group_id"] is not None for g in gathered)
        if has_gid:
            if not all(g["group_id"] is not None for g in gathered):
                raise RuntimeError("Some ranks have group_id while others do not.")
            group_id = np.concatenate([g["group_id"] for g in gathered], axis=0)[:total_samples]
        else:
            group_id = None

        return y_pred, y_true, group_id

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            # Change: Each rank displays its own prediction progress
            if self._verbose > 0:
                data_generator = tqdm(
                    data_generator,
                    disable=False,
                    file=sys.stdout,
                    desc=f"Rank {self.rank} | Predict",
                    position=self.rank,
                    leave=(self.rank == 0),
                    dynamic_ncols=True
                )

            amp_ctx = self._get_amp_context()
            with amp_ctx:
                for batch_data in data_generator:
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].float().data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
        torch.save(self.state_dict(), checkpoint)

    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def delete_checkpoint(self):
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)
            logging.info("Deleted model checkpoint: {}".format(self.checkpoint))
        else:
            logging.info("Model checkpoint already removed or not found: {}".format(self.checkpoint))

    def get_output_activation(self, task):
        if task == "binary_classification":
            return Float32Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        embedding_params = 0
        dense_params = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            param_num = param.numel()

            if "embedding" in name:
                embedding_params += param_num
                if count_embedding:
                    total_params += param_num
            else:
                dense_params += param_num
                total_params += param_num

        if self._is_main_process():
            logging.info("Total number of parameters: {}.".format(total_params))
            logging.info("Number of embedding parameters: {}.".format(embedding_params))
            logging.info("Number of dense parameters: {}.".format(dense_params))
