# LongMix-Rec

基于 [UniRank](https://github.com/salmon1802/UniRank) 的推荐排序学习项目，使用 KuaiRand-27K，研究长行为序列建模与特征交互的组合。训练入口、分块数据加载、六任务评估和最佳权重保存共用 UniRank 框架。

## 模型与实验

| 实验 ID | 结构 |
| --- | --- |
| `LONGER_KuaiRand_Video_Action` | 分组历史编码、最近历史 query、候选感知 global query |
| `RankMixer_KuaiRand_Video_Action` | 候选感知历史池化 → token 化 → Token Mixing + Per-Token FFN |
| `TokenMixer_KuaiRand_Video_Action` | 候选感知历史池化 → global/group token → 两阶段交互 |
| `LONGER_RankMixer_NoMixer_KuaiRand_Video_Action` | LONGER → masked mean → 特征拼接 → Chunk Tokenizer → 六任务塔 |
| `LONGER_RankMixer_Dense_KuaiRand_Video_Action` | 上述结构加入一层 dense RankMixer |
| `LONGER_RankMixer_KuaiRand_Video_Action` | 上述结构加入一层 MoE RankMixer，每个 token 四个专家 |

实验参数见 [model_config.yaml](config/model_config.yaml)。组合模型的三个实验共享 LONGER 编码器和预测塔配置，便于进行少量对照实验。建议按 NoMixer → Dense → MoE 的顺序运行。

MoE 使用两条共享专家的完整路径：训练时分别执行 Softmax train 路由和 ReLU infer 路由，六任务损失取两路平均，再加 infer 路由的 L1 正则；验证和推理只使用 infer 路由。正则分支仅更新 infer 路由器，任务梯度仍传给路由器、专家和上游模型。

当前 MoE **仍计算全部专家**，ReLU 产生门控稀疏性，尚未实现跳过零门控专家的稀疏 dispatch，因此不能据此宣称推理 FLOPs 或延迟下降。TokenMixer 基线中的 `PerTokenSwiGLU` 为沿用的类名，当前实际使用 GELU 门控（GEGLU 形式）。项目包含学习用途的模型改造，不宣称完整复现相关论文的工业系统或指标。

## 环境安装

本次检查环境为 Python 3.11、PyTorch 2.8.0 + CUDA 12.6、Windows、RTX 4070 12GB。目标部署环境为 Linux 单卡 A800 80GB；该服务器和全量训练尚未实测。

在服务器创建环境并进入仓库：

```bash
git clone https://github.com/itszqqqq/LongMix-Rec.git
cd LongMix-Rec
conda create -n longmix python=3.11 -y
conda activate longmix
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

上面固定为已检查的 PyTorch wheel；服务器驱动需支持对应 CUDA runtime。CPU 检查可安装 CPU wheel，并在运行时传入 `--gpu -1 --no-enable_bf16`。默认关闭 `torch.compile`，BF16 默认开启；`--enable_bf16` / `--no-enable_bf16` 可以覆盖 YAML 配置。

## 数据准备

已处理的数据可直接传到服务器的 `data/KuaiRand_Video_Action/`：

```text
data/KuaiRand_Video_Action/
  meta_data.json
  train/{data,user_info,item_info}/part-00000.parquet ...
  valid/{data,user_info,item_info}/part-00000.parquet ...
  test/{data,user_info,item_info}/part-00000.parquet ...
```

同一 split 的三个目录必须按 `part-xxxxx` 配对。`user_index/item_index` 是块内索引，embedding 使用恢复后的全局 ID。词表配置见 [dataset_config.yaml](config/dataset_config.yaml)，必须与数据的 `meta_data.json` 一致。数据、环境和训练权重不随 Git 仓库上传。

从原始 CSV 处理时，先下载解压到 `data/KuaiRand/KuaiRand-27K/`（下载命令见 [download.sh](data/KuaiRand/download.sh)），再在项目根目录执行：

```bash
python data/KuaiRand/preprocess_Kuairand_seq_action.py --data_dir ./data/KuaiRand/KuaiRand-27K/data --output_dir ./data/KuaiRand_Video_Action --train_blocks 32 --valid_blocks 8 --test_blocks 8 --num_workers 4
```

预处理按日期划分训练、验证、测试集。加载时依据 `seq_len` 取目标之前的历史前缀，再截取最近 `max_len` 条、左侧补零；候选位置的 action 固定为 0，避免当前反馈标签泄漏。六任务为 click、follow、like、comment、forward、long_view。时间戳虽然保存在 side info 中，当前模型没有使用时间间隔 embedding。

`rebuild_dataset: false` 表示 Parquet 已编码；训练启动时建立 feature map，无需为千万级 ID 构建 Python token 字典。实际数据共有 32/8/8 个训练/验证/测试块，合计约 3.23 亿样本。加载器按整块读入，并缓存 side info，CPU 内存也需要关注；首次服务器运行可将 `num_workers` 设为 0、`block_cache_size` 设为 1，再按内存情况增加。

## 检查与训练

先用微型数据走通正式训练入口，不需要下载数据集：

```bash
python -m unittest discover -s tests -v
```

测试覆盖四种模型及组合模型的 NoMixer / Dense / MoE、六个 YAML 实验入口、训练和评估、保存加载、尾部梯度累积、padding/因果隔离、MoE 梯度；有 CUDA 时自动执行 BF16 和激活重算检查。

运行组合模型的三个对照实验：

```bash
python run_expid.py --expid LONGER_RankMixer_NoMixer_KuaiRand_Video_Action --gpu 0
python run_expid.py --expid LONGER_RankMixer_Dense_KuaiRand_Video_Action --gpu 0
python run_expid.py --expid LONGER_RankMixer_KuaiRand_Video_Action --gpu 0
```

单次后台运行示例：

```bash
mkdir -p logs
nohup python -u run_expid.py --expid LONGER_RankMixer_KuaiRand_Video_Action --gpu 0 > logs/moe.console.log 2>&1 &
tail -f logs/moe.console.log
```

`run_all.sh` 在 Bash 下依次执行三个独立基线。组合模型默认 `batch_size=128, accumulation_steps=8, epochs=1`，完整窗口有效 batch 为 1024；最后不足累积窗口的梯度也会在验证前更新。独立基线默认 batch 为 8192，首次运行可先改小。历史长度须被 `num_groups` 整除，`query_size` 不超过分组后长度，注意力维度须被头数整除。

## 结果输出与验证范围

结果保存在：

```text
checkpoints/KuaiRand_Video_Action/<experiment_id>.log
checkpoints/KuaiRand_Video_Action/<experiment_id>.model
```

日志包含六任务 AUC/logloss 和任务均值；默认监控平均 AUC。训练结束会加载最佳权重并自动评估 test，权重继续保留。验证和测试保留每块最后不足一个 batch 的样本。相同实验 ID 再次运行会覆盖同名日志和权重，保存实验结果时应使用不同 ID 或提前归档。

保存的是模型 state dict，包含两条路由器和共享专家；尚未实现带优化器状态的断点续训。检查结果见 [验证记录](docs/验证记录.md)。微型数据和真实小切片验证用于确认流程、数值和接口；全量词表的显存、A800 吞吐、最终指标和完整原始 CSV 预处理仍需在服务器实测。

## 代码入口

- [LONGER_RankMixer.py](model_zoo/LONGER_RankMixer.py)：组合模型、masked readout、双路任务损失。
- [LONGER.py](model_zoo/LONGER.py)：分组历史编码、padding 和组间因果约束。
- [RankMixer.py](model_zoo/RankMixer.py)：特征交互与 dense/MoE FFN 选择。
- [per_token_ffn.py](unirank/pytorch/layers/interactions/per_token_ffn.py)：专家、Softmax/ReLU 路由和 L1 正则。
- [unirank_dataloader.py](unirank/pytorch/dataloaders/unirank_dataloader.py)：分块索引恢复与历史截取。
- [rank_model.py](unirank/pytorch/models/rank_model.py)、[multitask_model.py](unirank/pytorch/models/multitask_model.py)：训练、优化器、六任务评估。

## 致谢与许可

项目基于 UniRank，保留原有代码来源和 Apache-2.0 许可，详见 [LICENSE](LICENSE)。数据集使用遵循 KuaiRand 原始说明与许可。
