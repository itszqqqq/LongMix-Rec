# LongMix-Rec

基于 [UniRank](https://github.com/salmon1802/UniRank) 的推荐排序学习项目，使用 KuaiRand-27K，研究长行为序列建模与特征交互的组合。本版保存可运行的 **dense 基线**，便于后续做少量实验和整理面试讲述。

## 当前模型

| 配置中的 model | 主要结构 |
| --- | --- |
| `LONGER` | 历史分组、组内建模、最近历史 query 与全局 query |
| `RankMixer` | 候选感知的历史池化 → 特征 token 化 → Token Mixing + Per-Token FFN |
| `TokenMixer` | 候选感知的历史池化 → global/group token → 两阶段交互 |
| `LONGER_RankMixer` | LONGER 编码 → masked mean → 拼接用户/候选特征 → Chunk Tokenizer → dense RankMixer → 六任务预测 |

`LONGER_RankMixer_NoMixer_KuaiRand_Video_Action` 将交互层数设为 0，作为同一 LONGER 编码器下的交互消融。各实验定义见 [model_config.yaml](config/model_config.yaml)。

本版本没有接入 ReLU Routing 或 DTSI-MoE。`per_token_ffn.py` 中保留的旧 `SparseMoELayer` 是未使用的草稿，不属于可运行模型接口。本项目也不宣称复现论文中的全部工业实验或性能。

## 环境

已验证：Python 3.11、PyTorch 2.8.0 + CUDA 12.6、Windows、RTX 4070 12GB。完整训练的目标环境为单卡 A800 80GB，尚未完成该环境的全量验证。

创建并激活虚拟环境后，在项目根目录安装依赖：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

CUDA wheel 选择可参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v280)。仅做 CPU 检查时，可将第一条命令的索引改为 `https://download.pytorch.org/whl/cpu`。

默认关闭 `torch.compile`，先使用普通 PyTorch 跑通；后续可在支持的环境中自行打开 `enable_torch_compile`。BF16 默认开启，命令行 `--no-enable_bf16` 可以关闭，`--enable_bf16` 可以开启；未指定时遵循 YAML 配置。

## 数据

原始数据下载入口在 [download.sh](data/KuaiRand/download.sh)。已经有处理好的数据时，直接检查目录，不需要重跑预处理：

```text
data/KuaiRand_Video_Action/
  meta_data.json
  train/{data,user_info,item_info}/part-00000.parquet ...
  valid/{data,user_info,item_info}/part-00000.parquet ...
  test/{data,user_info,item_info}/part-00000.parquet ...
```

同一个 split 内，三个目录的 `part-xxxxx` 必须配对。`user_index/item_index` 是块内局部索引，embedding 使用恢复后的全局 `user_id/item_id`；不同块不能混用 side info。词表大小与特征定义见 [dataset_config.yaml](config/dataset_config.yaml)，应与该份数据的 `meta_data.json` 一致。

从原始 CSV 重新处理时，在根目录运行：

```bash
python data/KuaiRand/preprocess_Kuairand_seq_action.py --data_dir ./data/KuaiRand/KuaiRand-27K/data --output_dir ./data/KuaiRand_Video_Action --train_blocks 32 --valid_blocks 8 --test_blocks 8 --num_workers 4
```

预处理按日期划分训练/验证/测试集，并保存完整用户轨迹。加载器使用 `seq_len` 取目标之前的历史，截取最近 `max_len` 条、左侧补零；候选位置的 action 固定为 0，避免将当前反馈标签作为输入。六个任务为 click、follow、like、comment、forward、long_view。

`rebuild_dataset: false` 表示 Parquet 已完成编码。启动训练时只从配置建立 `feature_map.json`，不再为 3200 万级 ID 重建 Python 词表。时间戳虽在 side info 中保存，但当前模型没有使用时间间隔 embedding。

## 运行

先运行不依赖真实数据集的检查：

```bash
python -m unittest discover -s tests -v
```

再启动主模型：

```bash
python run_expid.py --expid LONGER_RankMixer_KuaiRand_Video_Action --gpu 0
```

做交互消融：

```bash
python run_expid.py --expid LONGER_RankMixer_NoMixer_KuaiRand_Video_Action --gpu 0
```

独立基线可替换实验名为 `LONGER_KuaiRand_Video_Action`、`RankMixer_KuaiRand_Video_Action`、`TokenMixer_KuaiRand_Video_Action`。`run_all.sh` 在 Linux/Bash 下依次运行这三个独立基线。CPU 使用 `--gpu -1 --no-enable_bf16`；默认 GPU 参数现在为单卡 `0`。

训练采用 dense AdamW + sparse embedding Adagrad，支持梯度累积、BF16 和可选的激活重算。主模型配置为 `batch_size=128, accumulation_steps=8`，完整窗口的有效 batch 为 1024；最后不足一个窗口的 batch 也会在验证前完成更新。序列长度应能被 `num_groups` 整除，`query_size` 不超过分组后的历史长度，`token_dim` 能被注意力头数整除。

日志和最佳权重保存在 `checkpoints/<dataset_id>/<experiment_id>.log/.model`。训练结束自动加载最佳权重、评估测试集，并保留权重。验证和测试保留各块最后不足一个 batch 的样本。保存的是模型权重，当前没有实现包含优化器状态的断点续训。

## 验证范围

本版检查记录见 [基线验证说明](docs/基线验证说明.md)。微型数据测试覆盖四种模型的 CPU 训练、六任务评估、保存/加载一致性，CUDA BF16 与激活重算，以及全部五个 YAML 实验入口。

另外检查了现有真实数据的 48 组三文件块的字段、配对、行数和词表 metadata，并用真实数据的小切片完成四个模型的 CUDA BF16 参数更新。切片测试压缩了 item ID 词表、缩小了模型尺寸，**不代表全词表训练显存和吞吐已经验证**。

全量预处理重跑、A800 全量训练、多卡 DDP、论文指标及吞吐对比均未在本次检查中验证。单个真实训练块有约八百万行，加载器按整块读取；CPU 内存、worker 数量和缓存数量也会影响资源使用。先使用主模型的小 batch 配置，再依据实际显存和内存调整。

## 代码入口

- [LONGER_RankMixer.py](model_zoo/LONGER_RankMixer.py)：组合模型与 masked readout。
- [LONGER.py](model_zoo/LONGER.py)：长序列编码、padding 和因果约束。
- [RankMixer.py](model_zoo/RankMixer.py)：dense 特征交互。
- [unirank_dataloader.py](unirank/pytorch/dataloaders/unirank_dataloader.py)：分块数据拼接和历史截取。
- [rank_model.py](unirank/pytorch/models/rank_model.py)、[multitask_model.py](unirank/pytorch/models/multitask_model.py)：训练循环、优化器和六任务评估。

## 致谢与许可

本项目基于 UniRank，保留其代码来源与 Apache-2.0 许可，详见 [LICENSE](LICENSE)。数据集使用请遵循 KuaiRand 的原始说明与许可。
