#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_Kuairand_seq_action.py — KuaiRand-27K memory-efficient blocked preprocessing
================================================================================================
Process data in batches and write TAAC2025-compatible blocks to limit peak memory.

Split strategy (8:1:1 by date):
  - Sort all dates chronologically.
  - Use the earliest ~80% for training, the next ~10% for validation, and the latest ~10% for testing.

Processing stages:
  Phase 1 — Convert CSV files into user-hash-partitioned Parquet files.
  Phase 2 — Encode and split each partition, then write block data and user_info.
  Phase 3 — Build item_info, write metadata and the block manifest, and remove temporary files.

New features:
  1. Storage in blocks (consistent with TAAC2025):
     - Divide each split into multiple blocks.
     - Write separate data, user_info, and item_info files per block.
     - Remap user_index and item_index to contiguous block-local IDs.
     - Keep user_id and item_id global for consistent embeddings.
  2. OOV low-frequency feature filtering (min_feat_count=2):
     - Count feature values while building vocabularies.
     - Map values occurring fewer than min_feat_count times to 0.
     - Filter only USER_STATIC_FEATURES and ITEM_STATIC_FEATURES.
     - Preserve global user_id, item_id, and action mappings.

Notes for this release:
  - Do not build user_info.behavior_type_mask.
  - Store action_vocab in meta_data.json.
  - Store user_info.full_timestamp_seq.
  - Let the dataloader derive task-specific masks from full_action_seq.

Dependencies:
    pip install pandas numpy pyarrow
"""

import argparse
import gc
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
#  1. Constants and column definitions
# ================================================================

# OOV filtering threshold: feature values occurring fewer than this threshold are mapped to 0 (unknown/padding)
MIN_FEAT_COUNT = 2

LOG_FILES = [
    "log_random_4_22_to_5_08_27k.csv",  # 随机曝光日志
    "log_standard_4_08_to_4_21_27k_part1.csv",   # 标准推荐日志
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
]

LOG_LOAD_COLUMNS = [
    "user_id", "video_id", "time_ms",
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
    "play_time_ms", "tab",
]

LOG_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "time_ms": "int64",
    "is_click": "int8",
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "long_view": "int8",
    "play_time_ms": "int32",
    "tab": "int8",
}

LABEL_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
]

USER_STATIC_FEATURES = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]
# user_features_27k.csv: 26个用户画像字段

CONTEXT_FEATURES = ["tab", "day_of_week", "is_weekend", "hour"]

ITEM_STATIC_FEATURES = ["video_type", "primary_tag", "music_type", "duration_bucket"]
# video_features_basic_27k.csv: 视频类型，首个标签，音乐类型、视频时长
# video_features_statistic_27k.csv: 不读取

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id"]
    + USER_STATIC_FEATURES + CONTEXT_FEATURES + LABEL_COLUMNS
)


# ================================================================
#  2. Utilities
# ================================================================

def build_ordered_vocab(values, start=1):
    """Construct vocab from iterable object, 0 is reserved for padding/unknown when start=1."""
    uniq = list(dict.fromkeys(str(v) for v in values))  # 把每个值统一转换成字符串; 去掉重复值, 保留首次出现的顺序。
    return {v: i for i, v in enumerate(uniq, start=start)}


PLAY_TIME_BINS = [-1, 0, 1000, 3000, 7000, 18000, 60000, float("inf")]     # 播放时长区间（毫秒）
PLAY_TIME_LABELS = ["0", "0-1s", "1-3s", "3-7s", "7-18s", "18-60s", "60s+"]  # 播放标签

DURATION_BINS = [-1, 0, 3000, 7000, 15000, 30000, 60000, float("inf")]   # 视频时长区间（毫秒）
DURATION_LABELS = ["0", "0-3s", "3-7s", "7-15s", "15-30s", "30-60s", "60s+"]  # 视频标签


def bucket_play_time(s: pd.Series) -> pd.Series:
    # 按用户播放时长分桶
    return pd.cut(s.fillna(0), bins=PLAY_TIME_BINS, labels=PLAY_TIME_LABELS).astype(str)  # pd.cut() 默认使用左开右闭区间
    # s = pd.Series([0, 1000, 1001, 3000, None])
    # bucket_play_time(s) -> ["0", "0-1s", "1-3s", "1-3s", "0"]

def bucket_duration(s: pd.Series) -> pd.Series:
    # 按视频时长分桶
    return pd.cut(s.fillna(0), bins=DURATION_BINS, labels=DURATION_LABELS).astype(str)

def fmt_date_int(d: int) -> str:
    # 整数日期变成可读字符串
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def build_date_split(sorted_dates, train_ratio, valid_ratio, test_ratio):
    '''
    根据时间顺序，确定哪些日期用于训练、验证和测试。 
    返回日期列表及切分边界，实际的样本行切分由后面的 process_partition() 完成
    '''
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError(
            f"There are only {n_days} different dates in the data, and it takes at least 3 days to split train/valid/test by date."
        )

    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()

    raw = ratios * n_days
    counts = np.floor(raw).astype(int)
    rem = n_days - counts.sum()

    if rem > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(rem):
            counts[order[i % 3]] += 1

    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                break
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1

    n_train, n_valid, n_test = int(counts[0]), int(counts[1]), int(counts[2])

    train_dates = sorted_dates[:n_train]
    valid_dates = sorted_dates[n_train:n_train + n_valid]
    test_dates = sorted_dates[n_train + n_valid:]

    return {
        "n_days": n_days,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "valid_start_date": valid_dates[0],
        "test_start_date": test_dates[0],
    }


# ================================================================
#  3. Vocab & mapping construction (does not rely on log scanning)
# ================================================================

def load_user_features(data_dir: Path) -> pd.DataFrame:
    """Loading user features (27K lines, small). 读取并清理用户画像"""
    fp = data_dir / "user_features_27k.csv"
    print(f"  [Load] {fp.name}")
    cols = ["user_id"] + USER_STATIC_FEATURES
    uf = pd.read_csv(fp, usecols=cols)
    uf["user_id"] = uf["user_id"].astype("int32")
    for col in USER_STATIC_FEATURES:
        uf[col] = uf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    print(f"         {len(uf):,} users")
    return uf


def build_all_vocabs(uf: pd.DataFrame) -> dict:
    """Build all category feature vocabs (0=padding/unknown) from user feature table + known domain.
    Note: Only static vocab is built here, and filtering based on frequency is not performed.
    OOV filtering is done in build_vocabs_with_oov_filter."""
    vocabs = {}
    # 参数 uf 是用户画像表，一行一个用户。函数逐个处理 USER_STATIC_FEATURES 中的字段
    for col in USER_STATIC_FEATURES:
        vals = sorted(set(uf[col].unique()) | {"__MISSING__"})  # 取出该列所有唯一值转换为集合，并加入 __MISSING__ 作为缺失值占位符
        vocabs[col] = {v: i + 1 for i, v in enumerate(vals)}    # 将类别映射为从 1 开始的整数，0 暂不分配
    # 如 vocabs[col] = {"A": 1, "B": 2, "C": 3, "__MISSING__": 4}
    
    # 为上下文字段建立固定词表
    vocabs["tab"] = {str(i): i + 1 for i in range(15)}
    vocabs["day_of_week"] = {str(i): i + 1 for i in range(7)}
    vocabs["is_weekend"] = {"0": 1, "1": 2}
    vocabs["hour"] = {str(i): i + 1 for i in range(24)}
    vocabs["play_time_bucket"] = {l: i + 1 for i, l in enumerate(PLAY_TIME_LABELS)}
    return vocabs


def build_vocabs_with_oov_filter(
    uf: pd.DataFrame,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """Build a vocab based on frequency statistics and filter feature values occurring fewer than min_feat_count times.
    根据交互频次，过滤画像类别并重新编号
    Args:
        uf: User Features DataFrame (USER_STATIC_FEATURES is in string form)
        feat_value_counters: {feature_name: {value_int: count}}
            The frequency of encoded int values in uf_enc
        min_feat_count: frequency threshold, feature values lower than this value will not be added to vocab

    Returns:
        vocabs: {feature_name: {value_str: int_id}}
        - ID columns (user_id, item_id, action) are not processed here
        - USER_STATIC_FEATURES / ITEM_STATIC_FEATURES will be filtered based on frequency
        - Other fixed fields (tab, day_of_week, is_weekend, hour, play_time_bucket) are not filtered

    Notes:
        Since uf_enc is encoded int16 based on the initial vocab, so is the key in feat_value_counters
        This encoded int value. We use this int value as the vocab key instead of a string.
        The final output vocab is in the form of {int_value_str: int_id}.
    """
    vocabs = {}

    # ---- USER_STATIC_FEATURES: Filter based on frequency ----
    for col in USER_STATIC_FEATURES:
        vocab = {}
        idx = 1
        # uf_enc has been encoded as int16, but here we directly use the int value as the key
        # Collect all occurrences of int values (including 0 for __MISSING__)
        if col in feat_value_counters:
            counter = feat_value_counters[col]
            # 0 means __MISSING__, always reserved
            all_int_vals = set(counter.keys()) | {0}
            for v in sorted(all_int_vals):
                cnt = int(counter.get(v, 0))
                # 0 (MISSING) is always retained; other values are filtered if frequency < min_feat_count
                if v != 0 and cnt > 0 and cnt < min_feat_count:
                    continue
                vocab[v] = idx
                idx += 1
        else:
            # Features without frequency statistics are all retained.
            vocab[0] = 1
            idx = 2
        vocabs[col] = vocab

    # ---- Fixed domain: No filtering ----
    vocabs["tab"] = {str(i): i + 1 for i in range(15)}
    vocabs["day_of_week"] = {str(i): i + 1 for i in range(7)}
    vocabs["is_weekend"] = {"0": 1, "1": 2}
    vocabs["hour"] = {str(i): i + 1 for i in range(24)}
    vocabs["play_time_bucket"] = {l: i + 1 for i, l in enumerate(PLAY_TIME_LABELS)}

    return vocabs


def build_item_vocabs_with_oov_filter(
    item_feat_df: pd.DataFrame,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """Construct item feature vocab to filter low-frequency values based on frequency.
    为每个视频属性建立词表，并排除部分低频取值

    Args:
        item_feat_df: item feature DataFrame (cleaned to str)
        feat_value_counters: {feature_name: {value_str: count}}
        min_feat_count: frequency threshold

    Returns:
        vocabs: {feature_name: {value_str: int_id}}
    """
    vocabs = {}
    for col in ITEM_STATIC_FEATURES:
        vocab = {}
        idx = 1
        all_vals = set(item_feat_df[col].unique()) | {"__MISSING__"}
        if col in feat_value_counters:
            counter = feat_value_counters[col]
            for v in sorted(all_vals):
                cnt = int(counter.get(v, 0))
                if cnt > 0 and cnt < min_feat_count:
                    continue
                vocab[v] = idx
                idx += 1
        else:
            for v in sorted(all_vals):
                vocab[v] = idx
                idx += 1
        vocabs[col] = vocab
    return vocabs


def build_action_maps():
    """Enumerate all 2^6=64 action patterns → (pattern_int→name, name→code).
    把六个反馈标签组合成一个行为类别"""
    pat2name = {}
    for p in range(64):
        if p == 0:
            pat2name[p] = "exposure"
        else:
            parts = [c for i, c in enumerate(LABEL_COLUMNS) if p & (1 << i)]
            pat2name[p] = "|".join(parts)
            # 如 pat2name[3] = "is_click|is_like"
    # 将这些名称排序，再分配 1～64 的行为编号，保留编号 0 给 padding 等用途
    name2code = {n: i + 1 for i, n in enumerate(sorted(set(pat2name.values())))}
    return pat2name, name2code


def encode_user_features_to_int(uf: pd.DataFrame, vocabs: dict) -> pd.DataFrame:
    """Precode user features into int16, which greatly saves memory when merging into logs.
    根据每个用户属性的词表，把属性值转换成整数编号，返回一张新的用户特征表"""
    uf_enc = uf[["user_id"]].copy()
    for col in USER_STATIC_FEATURES:
        uf_enc[col] = uf[col].map(vocabs[col]).fillna(0).astype("int16")
    return uf_enc


# ================================================================
#  4. Phase 1: CSV → Partitioned Parquet
# ================================================================

def _preprocess_chunk(
    chunk: pd.DataFrame,
    uf_enc: pd.DataFrame,
    vocabs: dict,
):
    """
    Execute for a single chunk (~2M lines):
      1. Timestamp decomposition (retain time_ms for subsequent sorting and user_info.full_timestamp_seq)
      2. Extract the date column (int32, YYYYMMDD) for segmentation by date
      3. play_time bucketing + context feature encoding → int8
      4. Merge precoded user features → int16
    _preprocess_chunk() 没有这条 merge 操作，也没有使用 uf_enc 的语句。所以它只处理日志中的时间、播放时长、上下文字段，返回的表里不会因此出现用户画像列
    注释写了“合并用户特征”，但这个操作实际发生在后面的函数中。画像合并安排在 Phase 2 的 process_partition()。

    Note: This function is shared between Phase 1 and Phase 2. When Phase 1 is called
    vocabs may be an empty dictionary (in this case, str feature values will be retained first), in Phase 2
    vocabs has been constructed.

    However, to simplify the implementation, Phase 1 only performs timestamp and context encoding when calling this function.
    Don't do USER_STATIC_FEATURES encoding (because ultimately vocab depends on OOV filtering).
    """
    # ---- Timestamp decomposition ----
    dt = pd.to_datetime(chunk["time_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")  # 将时间戳转换为北京时间
    # 提取星期、周末标识、小时和日期
    chunk["day_of_week"] = dt.dt.dayofweek.astype("int8")  # 0=Monday, 6=Sunday
    chunk["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")  # 0=weekday, 1=weekend
    chunk["hour"] = dt.dt.hour.astype("int8")  # 0-23

    # ---- Extract date column (YYYYMMDD int32) ----
    chunk["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("int32")  # 整数日期用于后续划分训练、验证、测试集
    del dt

    # ---- play_time bucketing + encoding ----
    # 将实际播放时长分桶并编码
    pt_bucket_str = bucket_play_time(chunk["play_time_ms"])  # 先把毫秒数转换成区间名称
    chunk["play_time_bucket"] = (
        pt_bucket_str.map(vocabs["play_time_bucket"]).fillna(0).astype("int8")
    )
    del pt_bucket_str

    # ---- Context feature encoding ----
    # 将上下文属性映射为类别编号
    chunk["tab"] = (
        chunk["tab"].astype(str).map(vocabs["tab"]).fillna(0).astype("int8")
    )
    chunk["day_of_week"] = (
        chunk["day_of_week"].astype(str).map(vocabs["day_of_week"]).fillna(0).astype("int8")
    )
    chunk["is_weekend"] = (
        chunk["is_weekend"].astype(str).map(vocabs["is_weekend"]).fillna(0).astype("int8")
    )
    chunk["hour"] = (
        chunk["hour"].astype(str).map(vocabs["hour"]).fillna(0).astype("int8")
    )

    # ---- Discard columns no longer needed ----
    # 已经生成了 play_time_bucket，这里删除原始播放时长列。inplace=True 表示直接修改当前 chunk
    chunk.drop(columns=["play_time_ms"], inplace=True)

    # ---- Remove missing ID rows ----
    # 如果一条记录的用户 ID 或视频 ID 缺失，就删除整条记录，因为后续无法可靠地关联用户、视频信息
    chunk.dropna(subset=["user_id", "video_id"], inplace=True)

    return chunk


def phase1_partition_to_parquet(
    data_dir: Path,   # 原始 CSV 日志所在目录
    tmp_dir: Path,    # 临时分区 Parquet 的存放目录
    uf_enc: pd.DataFrame,   # 预编码的用户画像表
    vocabs: dict,   # 初始词表
    n_parts: int,   # 用户分区数量
    chunk_size: int,   # 每次读取 CSV 的行数
    buffer_flush_size: int,  # 当缓冲区累计的行数达到这个阈值时，就将缓冲区内容写入 Parquet 文件
):
    """
    Read CSV file-by-file and chunk-by-chunk → preprocess → write to Parquet partitioned by user_id hash.
    Simultaneous statistics:
      - user_counts / item_ids / all_dates
      - feat_value_counters: {feature_name: {value_int: count}}
        Used for subsequent OOV low-frequency feature filtering
    分批读取原始日志，处理每批记录，再按用户 ID 分区保存成临时 Parquet, 同时统计用户交互次数、出现过的视频 ID 和日期
    （同一个用户无论出现在什么 CSV、什么读取批次中，都会进入同一个分区。后续才能在一个分区内收集这个用户的完整交互，并按时间整理历史。
      一个分区可以包含多个用户，包含多个 parquet 文件）

    Note: To support OOV filtering, Phase 1 does not do the final encoding of USER_STATIC_FEATURES here.
    Instead, it only does temporal/contextual feature encoding and retains the user_id for subsequent merges.
    The frequency statistics of USER_STATIC_FEATURES are calculated through the mapping of user_id and uf_enc.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    user_counts = defaultdict(int)  # 保存每个用户累计出现的次数
    item_ids = set()
    all_dates = set()

    # ---- Feature value frequency statistics container ----
    # USER_STATIC_FEATURES in uf_enc is already int16 encoding (based on initial vocab)
    # But the frequency should be based on "how many times the user appears in the log", so use user_counts * user feature value
    # Here is changed to: After Phase 1 is completed, use user_counts and uf_enc to jointly calculate
    feat_value_counters = None  # Placeholder, will be filled in the main process

    buffers = {p: [] for p in range(n_parts)}    # 存储每个分区的缓冲区数据块
    buf_sizes = {p: 0 for p in range(n_parts)}   # 存储每个分区缓冲区的累计行数
    file_counts = {p: 0 for p in range(n_parts)}  # 记录每个分区已经写入的 Parquet 文件数量

    def flush(pid):
        if not buffers[pid]:
            return
        # 将列表中的多张 DataFrame (chunk)按行拼接成一张表，ignore_index=True 会重新生成连续索引
        out = pd.concat(buffers[pid], ignore_index=True)
        fp = tmp_dir / f"part_{pid:03d}" / f"c_{file_counts[pid]:04d}.parquet"
        out.to_parquet(fp, index=False, engine="pyarrow")
        file_counts[pid] += 1
        buffers[pid] = []
        buf_sizes[pid] = 0
        del out

    # 逐个 CSV、逐个 chunk 地读取日志
    total_rows = 0
    for fname in LOG_FILES:
        print(f"  [Phase1] {fname}")
        reader = pd.read_csv(
            data_dir / fname,
            usecols=LOG_LOAD_COLUMNS,
            dtype=LOG_DTYPES,
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = _preprocess_chunk(chunk, uf_enc, vocabs)

            vc = chunk["user_id"].value_counts(sort=False)  # 统计当前 chunk 中每个用户出现多少次
            for uid, cnt in zip(vc.index, vc.values):
                user_counts[uid] += int(cnt)
            # 更新视频和日期集合
            item_ids.update(chunk["video_id"].unique().tolist())
            all_dates.update(chunk["date"].unique().tolist())

            part_arr = chunk["user_id"].values.astype(np.int64) % n_parts   # 分区编号 = 用户 ID 对分区数量取余
            for pid in range(n_parts):
                mask = part_arr == pid
                n_match = int(mask.sum())
                if n_match > 0:
                    buffers[pid].append(chunk.loc[mask].copy())
                    buf_sizes[pid] += n_match
                    if buf_sizes[pid] >= buffer_flush_size:
                        flush(pid)

            # 一个 chunk 处理结束后，更新总行数并清理临时变量
            total_rows += len(chunk)
            del chunk, part_arr
            gc.collect()

        print(f"         Accumulated {total_rows:,} rows")

    for pid in range(n_parts):
        flush(pid)

    print(
        f"  [Phase1] Completed: {total_rows:,} line,"
        f"{len(user_counts):,} user, {len(item_ids):,} video,"
        f"{len(all_dates)} different dates\n"
    )
    return dict(user_counts), item_ids, all_dates, feat_value_counters


def compute_feat_value_counters(
    user_counts: dict,
    uf_enc: pd.DataFrame,
) -> dict:
    """Calculate the frequency of feature values for USER_STATIC_FEATURES based on user_counts and uf_enc.
    统计每个用户静态属性的取值，在交互日志中累计出现了多少次，供后续 OOV 过滤使用
    The number of times each user appears in the log × user feature value = the total number of occurrences of the feature value in the log.

    Args:
        user_counts: {user_id: interaction_count}
        uf_enc: Precoded user features DataFrame (including user_id and USER_STATIC_FEATURES)

    Returns:
        {feature_name: {value_int: count}}
    """
    print("  [Phase1.5] Calculate USER_STATIC_FEATURES frequency based on user interaction times (for OOV filtering)")
    counters = {col: defaultdict(int) for col in USER_STATIC_FEATURES}

    # Convert user_counts to DataFrame
    uc_df = pd.DataFrame(
        list(user_counts.items()), columns=["user_id", "count"]
    )
    # Merge with uf_enc
    uc_df = uc_df.merge(uf_enc, on="user_id", how="left")

    for col in USER_STATIC_FEATURES:
        # Aggregate count by feature value
        grouped = uc_df.groupby(col)["count"].sum()  # 将属性编号相同的用户放在一组，累加这一组用户的交互次数。
        for v, cnt in grouped.items():
            counters[col][int(v)] += int(cnt)

    del uc_df
    return counters


# ================================================================
#  5. Phase 2: Per-partition Process + Incremental Write
# ================================================================
# 把 Phase 1 中按用户分区保存的临时日志，整理成“逐条预测样本＋用户完整历史轨迹”，然后按日期切分，并交给后续写入流程

def process_partition(
    tmp_dir: Path,   # Phase 1 临时目录
    pid: int,   # 要处理的分区编号
    valid_users: set,  # 通过最低交互次数过滤的用户集合 
    user_idx_map: dict,  # 原始用户 ID → 全局用户索引
    item_idx_map: dict,  # 原始视频 ID → 全局视频索引
    uf_enc: pd.DataFrame,  # 用户画像的中间整数编码表
    vocabs: dict,   # 过滤后的用户属性词表，中间编号 → 最终编号
    pat2name: dict,    # 六个反馈标签的组合编码映射
    name2code: dict,
    valid_start_date: int,   # 验证集、测试集的起始日期
    test_start_date: int,
):
    """
    To process a user partition:
      1. Read all parquet files under the partition
      2. Filter valid users
      3. Merge precoded user features → int16
      4. Sort by (user_id, time_ms)
      5. Encoding user_index / item_index / action / exposure
      6. Split train/valid/test by date
      7. Build user_info fragment (retain full_item_seq / full_action_seq / full_timestamp_seq)

    Returns dict{train, valid, test, user_info} or None.
    """
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))  # 获取分区目录下的所有 parquet 文件
    if not files:
        return None

    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[d["user_id"].isin(valid_users)]  # 只保留有效用户
        if len(d) > 0:
            dfs.append(d)
        del d
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)  # 合并所有 parquet 文件的内容，ignore_index=True 表示重新生成连续索引
    del dfs
    gc.collect()

    # ---- Merge precoded user features (int16) ----
    df = df.merge(uf_enc, on="user_id", how="left")    # 合并用户画像
    for col in USER_STATIC_FEATURES:
        df[col] = df[col].fillna(0).astype("int16")
        # Apply OOV filtering: values not in vocab are mapped to 0
        df[col] = df[col].map(vocabs[col]).fillna(0).astype("int32")

    # ---- Sort: by time within user ----
    df.sort_values(["user_id", "time_ms"], inplace=True)  # 将同一用户的记录放到一起，再按该用户的交互时间排列
    df.reset_index(drop=True, inplace=True)

    # ---- Encoding ID ----
    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    global_item_index = df["video_id"].map(item_idx_map)
    if global_item_index.isna().any():
        bad_video_ids = df.loc[global_item_index.isna(), "video_id"].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"partition={pid} There is video_id: {bad_video_ids} that has not entered the global item mapping"
        )
    df["item_index"] = global_item_index.astype(np.int32)
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)  # user_id 被覆盖为模型使用的用户类别编号，不再是原始用户 ID

    # ---- label → float32 ----
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(np.float32)  # 六个预测标签转换成浮点数，便于后续损失计算

    # ---- exposure + action ----
    binary = df[LABEL_COLUMNS].values.astype(np.int8)  # 六个标签的二进制矩阵，行数 = 样本数，列数 = 6
    df["exposure"] = (binary.sum(axis=1) == 0).astype(np.float32)  # 如果六个标签全为 0，则 exposure=1，否则 exposure=0

    pattern = np.zeros(len(df), dtype=np.int32)
    for i in range(len(LABEL_COLUMNS)):
        pattern += binary[:, i].astype(np.int32) << i   # << i 是左移运算，这里相当于乘以 2**i
    df["action"] = (
        pd.Series(pattern, index=df.index).map(pat2name).map(name2code).astype(np.int32)
    )  # 反馈组合整数 → 行为名称 → 最终 action 编号
    del binary, pattern

    # ---- Characteristic conversion to int32 ----
    for col in USER_STATIC_FEATURES + CONTEXT_FEATURES:
        df[col] = df[col].astype(np.int32)

    # ---- time_ms ensures int64 ----
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce").fillna(0).astype(np.int64)  # 确保时间戳是 int64 类型，避免后续排序或计算时出现溢出或精度问题

    # ---- seq_len ----
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)   # 给每个用户的交互记录按时间顺序编号，编号从 0 开始，表示该用户的历史交互长度

    # ---- Build user_info ----
    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].astype(int).tolist(),   # 该用户的完整视频交互序列
                "full_action_seq": gdf["action"].astype(int).tolist(),   # 该用户的完整行为序列
                "full_timestamp_seq": gdf["time_ms"].astype(np.int64).tolist(),  # 该用户的完整时间戳序列
            }
        )

    # ---- Split by date range ----
    date_col = df["date"]
    train_df = df[date_col < valid_start_date]
    valid_df = df[(date_col >= valid_start_date) & (date_col < test_start_date)]
    test_df = df[date_col >= test_start_date]

    # 负责选择预测样本最终保存的字段
    def _select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        present = [c for c in FINAL_COLUMNS if c in sdf.columns]
        return sdf[present].reset_index(drop=True)

    result = {
        "train": _select_final(train_df),
        "valid": _select_final(valid_df),
        "test": _select_final(test_df),
        "user_info": user_info_rows,
    }
    # 前三项是逐条样本表；最后一项是一行一个用户的完整轨迹信息

    del df, train_df, valid_df, test_df
    gc.collect()
    return result


_PHASE2_WORKER_STATE = {}

def _init_phase2_worker(
    tmp_dir,
    result_dir,
    valid_users,
    user_idx_map,
    item_idx_map,
    uf_enc,
    vocabs,
    pat2name,
    name2code,
    valid_start_date,
    test_start_date,
):
    """Initialize Phase 2 worker, save read-only state."""
    global _PHASE2_WORKER_STATE
    _PHASE2_WORKER_STATE = {
        "tmp_dir": Path(tmp_dir),
        "result_dir": Path(result_dir),
        "valid_users": valid_users,
        "user_idx_map": user_idx_map,
        "item_idx_map": item_idx_map,
        "uf_enc": uf_enc,
        "vocabs": vocabs,
        "pat2name": pat2name,
        "name2code": name2code,
        "valid_start_date": valid_start_date,
        "test_start_date": test_start_date,
    }


def _process_partition_to_temp(pid: int):
    """Process individual user partitions in parallel and write large results to temporary Parquet.
    多进程处理的辅助函数：处理指定分区，把结果保存为临时 Parquet，再将文件路径返回给主进程"""
    state = _PHASE2_WORKER_STATE
    result = process_partition(
        state["tmp_dir"],
        pid,
        state["valid_users"],
        state["user_idx_map"],
        state["item_idx_map"],
        state["uf_enc"],
        state["vocabs"],
        state["pat2name"],
        state["name2code"],
        state["valid_start_date"],
        state["test_start_date"],
    )
    if result is None:
        return {"pid": int(pid), "empty": True}

    part_result_dir = state["result_dir"] / f"part_{pid:03d}"  # 有数据时，创建该分区的临时结果目录
    if part_result_dir.exists():
        shutil.rmtree(part_result_dir)
    part_result_dir.mkdir(parents=True, exist_ok=True)

    user_info_rows = result["user_info"]
    user_info_file = part_result_dir / "user_info.parquet"
    pd.DataFrame(user_info_rows).to_parquet(user_info_file, index=False, engine="pyarrow")

    split_files = {}
    for split_name in ["train", "valid", "test"]:
        sdf = result[split_name]
        if len(sdf) == 0:
            continue
        split_file = part_result_dir / f"{split_name}.parquet"
        sdf.to_parquet(split_file, index=False, engine="pyarrow")
        split_files[split_name] = str(split_file)

    # 将文件位置返回给主进程
    payload = {
        "pid": int(pid),   # 分区编号
        "empty": False,    # 是否为空分区
        "result_dir": str(part_result_dir),  # 临时结果目录
        "user_info_file": str(user_info_file),    # 用户完整轨迹信息的 Parquet 文件
        "split_files": split_files,  # 逐条样本的 Parquet 文件位置，可能包含 train、valid、test 中的部分或全部
    }
    # 删除部分大型对象的局部引用，并尝试回收内存，最后返回路径信息
    del result, user_info_rows
    gc.collect()
    return payload


# ================================================================
#  6. Split Block Manager (refer to TAAC2025 implementation)
# ================================================================
# 这一部分的核心任务是：将已经划分好的 train、valid、test 数据，分别组织成多个可独立加载的 block，并为每个 block 配齐样本、用户历史和视频属性。
# 同时，它会建立紧凑的块内索引，让数据加载器能够在当前 block 内查表；用于模型 embedding 的全局特征编号则保持一致。

def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


class SplitBlockManager:
    """
    Divide a split (train/valid/test) into multiple blocks and write each block separately:
      - data/part-xxxxx.parquet
      - user_info/part-xxxxx.parquet
      - item_info/part-xxxxx.parquet

    inside block:
      - user_index / item_index remapped to block-local dense index
      - user_id / item_id maintains the global id and does not affect embedding consistency

    """

    def __init__(self, split_name: str, root_dir: Path, num_blocks: int):
        self.split_name = split_name
        self.root_dir = root_dir / split_name
        self.data_dir = self.root_dir / "data"
        self.user_info_dir = self.root_dir / "user_info"
        self.item_info_dir = self.root_dir / "item_info"

        self.num_blocks = int(num_blocks)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_info_dir.mkdir(parents=True, exist_ok=True)
        self.item_info_dir.mkdir(parents=True, exist_ok=True)

        self.data_writers = [None] * self.num_blocks  # 用于写入每个 block 的 ParquetWriter 对象
        self.user_writers = [None] * self.num_blocks  # 用于写入每个 block 的用户信息 ParquetWriter 对象

        self.block_rows = [0] * self.num_blocks   # 记录每个 block 已经写入的样本行数
        self.block_user_rows = [0] * self.num_blocks   # 记录每个 block 已经写入的用户历史行数

        self.partition_to_block = {}  # 记录每个 partition 被分配到了哪个 block

        self.user_maps = [dict() for _ in range(self.num_blocks)]   # 全局用户索引 → 当前块的本地用户索引，索引从 0 开始
        self.item_maps = [{0: 0} for _ in range(self.num_blocks)]   # 全局视频索引 → 当前块的本地视频索引，索引从 1 开始，0 → 0 留给 padding

        self.next_user_local = [0] * self.num_blocks  # 记录每个 block 下一个可用的本地用户索引
        self.next_item_local = [1] * self.num_blocks  # 记录每个 block 下一个可用的本地视频索引，1 开始，0 留给 padding

    def choose_block(self, n_rows: int) -> int:
        return int(np.argmin(self.block_rows))  # 选择当前样本行数最少的block

    def _get_or_add_user_local(self, bid: int, global_user_index: int) -> int:
        mp = self.user_maps[bid]   # 全局用户索引 → 当前块的本地用户索引
        if global_user_index not in mp:  # 如果当前全局用户索引还没有在这个块中出现过，就分配一个新的本地用户索引
            mp[global_user_index] = self.next_user_local[bid]  # 分配新的本地用户索引
            self.next_user_local[bid] += 1   # 更新下一个可用的本地用户索引
        return mp[global_user_index]

    def _get_or_add_item_local(self, bid: int, global_item_id: int) -> int:
        mp = self.item_maps[bid]
        if global_item_id not in mp:
            mp[global_item_id] = self.next_item_local[bid]
            self.next_item_local[bid] += 1
        return mp[global_item_id]

    # 将当前 DataFrame 追加到指定 block 的文件
    def _write_table(self, writer_list, bid: int, out_fp: Path, df: pd.DataFrame):
        if len(df) == 0:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)  # 将 DataFrame 转换为 PyArrow Table，preserve_index=False 表示不保留 DataFrame 的索引列
        if writer_list[bid] is None:
            writer_list[bid] = pq.ParquetWriter(str(out_fp), table.schema) # 创建 ParquetWriter 对象，指定输出文件路径和表的 schema，table.schema 描述各列的名称和类型
        writer_list[bid].write_table(table)

    def append_partition(
        self,
        pid: int,  # 来源用户 partition 的编号
        split_df: pd.DataFrame,  # 当前 partition 中属于当前 split 的样本表，包含 user_index、item_index、action、exposure 等列
        user_info_rows: list,   # 当前 partition 的用户完整轨迹信息，每个元素是一个字典，包含 user_index、full_item_seq、full_action_seq、full_timestamp_seq 等键
    ): 
        """Append the split data of a partition to a block.
        把一个 partition 中属于当前 split 的样本，整体追加到一个 block；
        同时将用户、视频的全局索引转换为块内索引，并写出样本表和用户历史表。

        Args:
            pid: partition id
            split_df: DataFrame of the split (including user_index, item_index, etc.)
            user_info_rows: list of dict (user_info from process_partition)
        """
        if len(split_df) == 0:
            return None

        bid = self.choose_block(len(split_df))   # 选择一个 block 来存放当前 partition 的数据，通常选择当前样本行数最少的 block
        self.partition_to_block[int(pid)] = int(bid)   # 记录当前 partition 被分配到了哪个 block

        # Build a lookup table of user_index -> user_info_row
        ui_lookup = {r["user_index"]: r for r in user_info_rows}

        active_users = pd.unique(split_df["user_index"].astype(np.int64))
        active_users_set = set(active_users.tolist())

        out_ui_rows = []   # 收集转换后的用户历史记录
        for g_user in active_users_set:
            if g_user not in ui_lookup:
                raise ValueError(
                    f"[{self.split_name}] partition={pid} user_index={g_user} "
                    f"Does not exist in user_info, logical exception."
                )
            row = ui_lookup[g_user]
            l_user = self._get_or_add_user_local(bid, int(g_user))   # 将全局用户索引转换为当前块的本地用户索引

            g_item_seq = row.get("full_item_seq", []) # 获取该用户的完整视频交互序列
            if not isinstance(g_item_seq, (list, tuple, np.ndarray)):
                g_item_seq = []
            l_item_seq = []

            for x in g_item_seq:
                g_item = safe_int(x, -1)  # 将全局视频索引转换为整数，如果无法转换则返回 -1
                if g_item <= 0:
                    raise ValueError(
                        f"[{self.split_name}] partition={pid} user_index={g_user} "
                        f"The full_item_seq contains the illegal global item_index={g_item}."
                    )
                l_item_seq.append(self._get_or_add_item_local(bid, g_item))  # 将全局视频索引转换为当前块的本地视频索引

            g_action_seq = row.get("full_action_seq", [])  # 获取该用户的完整行为序列
            if not isinstance(g_action_seq, (list, tuple, np.ndarray)):
                g_action_seq = []

            g_time_seq = row.get("full_timestamp_seq", [])  # 获取该用户的完整时间戳序列
            if not isinstance(g_time_seq, (list, tuple, np.ndarray)):
                g_time_seq = []

            out_ui_rows.append(
                {
                    "user_index": np.int32(l_user),  # 当前块的本地用户索引
                    "full_item_seq": [int(x) for x in l_item_seq],  # 将本地视频索引序列转换为整数列表
                    "full_action_seq": [int(x) for x in g_action_seq],  
                    "full_timestamp_seq": [int(x) for x in g_time_seq],
                }
            )

        out_ui_df = pd.DataFrame(out_ui_rows)  

        user_map = self.user_maps[bid]   # 全局用户索引 → 当前块的本地用户索引
        item_map = self.item_maps[bid]   # 全局视频索引 → 当前块的本地视频索引

        out_df = split_df.copy()
        out_df["user_index"] = out_df["user_index"].map(user_map).astype(np.int32)  # 将全局用户索引转换为当前块的本地用户索引
        local_item_index = out_df["item_index"].map(item_map)    # 将全局视频索引转换为当前块的本地视频索引
        if local_item_index.isna().any() or (local_item_index <= 0).any():  # 如果有任何视频索引在当前块中没有映射，或者映射后的本地索引小于等于 0（0 是 padding），则说明当前样本的历史映射不完整
            bad_global_items = (
                out_df.loc[local_item_index.isna() | (local_item_index <= 0), "item_index"]
                .drop_duplicates()
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"[{self.split_name}] partition={pid} The current sample has not written the complete historical mapping"
                f"global item_index: {bad_global_items}"
            )
        out_df["item_index"] = local_item_index.astype(np.int32)

        data_fp = self.data_dir / f"part-{bid:05d}.parquet"   # 当前块的样本表输出路径
        ui_fp = self.user_info_dir / f"part-{bid:05d}.parquet"   # 当前块的用户历史表输出路径

        self._write_table(self.data_writers, bid, data_fp, out_df)
        self._write_table(self.user_writers, bid, ui_fp, out_ui_df)

        self.block_rows[bid] += len(out_df)
        self.block_user_rows[bid] += len(out_ui_df)

        del out_df, out_ui_df, out_ui_rows
        gc.collect()
        return bid

    def close_writers(self):
        for w in self.data_writers:
            if w is not None:
                w.close()
        for w in self.user_writers:
            if w is not None:
                w.close()

    def write_item_info_blocks(self, global_item_lookup: pd.DataFrame):
        """Generate item_info.parquet for each block.

        Args:
            global_item_lookup: DataFrame indexed by global item_index,
                                Contains the ITEM_STATIC_FEATURES column
        """
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue

            item_map = self.item_maps[bid]
            local_size = max(item_map.values()) if len(item_map) > 0 else 0

            inv_global_item_id = np.zeros(local_size + 1, dtype=np.int32)
            for g_item_id, l_item_idx in item_map.items():
                inv_global_item_id[l_item_idx] = np.int32(g_item_id)

            out = {
                "item_index": np.arange(local_size + 1, dtype=np.int32),
                "item_id": inv_global_item_id.astype(np.int32),
            }

            if local_size > 0:
                gids = inv_global_item_id[1:].astype(np.int32)
                feat_df = global_item_lookup.reindex(gids).fillna(0)

                for col in ITEM_STATIC_FEATURES:
                    arr = np.zeros(local_size + 1, dtype=np.int32)
                    arr[1:] = feat_df[col].to_numpy(dtype=np.int32, copy=False)
                    out[col] = arr
            else:
                for col in ITEM_STATIC_FEATURES:
                    out[col] = np.zeros(1, dtype=np.int32)

            item_info_df = pd.DataFrame(out)
            out_fp = self.item_info_dir / f"part-{bid:05d}.parquet"
            item_info_df.to_parquet(out_fp, index=False, engine="pyarrow")

            del item_info_df
            gc.collect()

    def buildmanifest(self):
        blocks = []
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue
            blocks.append(
                {
                    "block_id": bid,
                    "rows": int(self.block_rows[bid]),
                    "users": int(len(self.user_maps[bid])),
                    "items": int(len(self.item_maps[bid]) - 1),
                    "data_file": str(self.data_dir / f"part-{bid:05d}.parquet"),
                    "user_info_file": str(self.user_info_dir / f"part-{bid:05d}.parquet"),
                    "item_info_file": str(self.item_info_dir / f"part-{bid:05d}.parquet"),
                    "source_partitions": [
                        int(pid) for pid, b in self.partition_to_block.items() if b == bid
                    ],
                }
            )
        return {
            "split": self.split_name,
            "num_blocks_configured": int(self.num_blocks),
            "num_blocks_written": int(len(blocks)),
            "blocks": blocks,
        }


def build_global_item_lookup(
    item_feat_df: pd.DataFrame,
    item_idx_map: dict,
) -> pd.DataFrame:
    """Build the global item lookup table (indexed by global item_index).

    Args:
        item_feat_df: item feature DataFrame (including video_id and ITEM_STATIC_FEATURES)
        item_idx_map: {video_id: global_item_index}

    Returns:
        DataFrame indexed by global_item_index, listed as ITEM_STATIC_FEATURES
    """
    vf = item_feat_df[item_feat_df["video_id"].isin(item_idx_map)].copy()
    vf["global_item_index"] = vf["video_id"].map(item_idx_map).astype(np.int32)
    vf = vf[["global_item_index"] + ITEM_STATIC_FEATURES].drop_duplicates(
        subset=["global_item_index"], keep="last"
    )
    vf = vf.set_index("global_item_index").sort_index()

    for col in ITEM_STATIC_FEATURES:
        vf[col] = pd.to_numeric(vf[col], errors="coerce").fillna(0).astype(np.int32)

    return vf

def load_video_basic_features(data_dir: Path) -> pd.DataFrame:
    """Load the basic video features and derive primary_tag + duration_bucket."""
    fp = data_dir / "video_features_basic_27k.csv"
    print(f"  [Load] {fp.name}")
    cols = ["video_id", "video_type", "music_type", "tag", "video_duration"]
    vf = pd.read_csv(fp, usecols=cols, dtype={"video_id": "int32"})
    vf["primary_tag"] = vf["tag"].astype(str).str.split(",").str[0].str.strip()
    vf["primary_tag"] = vf["primary_tag"].replace(
        {"nan": "__MISSING__", "": "__MISSING__"}
    )
    vf["duration_bucket"] = bucket_duration(vf["video_duration"].fillna(0))
    for col in ["video_type", "music_type"]:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    vf.drop(columns=["tag", "video_duration"], inplace=True)
    print(f"         {len(vf):,} videos")
    return vf


def build_item_info(data_dir: Path, item_idx_map: dict, output_dir: Path):
    """
    item_info.parquet (for non-blocked mode):
      - Loaded from video_features_basic_27k.csv
      - Only keep videos that appear in the log
      - Encoding ITEM_STATIC_FEATURES
      - Line 0 is padding

    In blocked mode, this function will not be called and SplitBlockManager.write_item_info_blocks will be used instead.
    """
    vf = load_video_basic_features(data_dir)

    vf = vf[vf["video_id"].isin(item_idx_map)].copy()
    vf["item_index"] = vf["video_id"].map(item_idx_map).astype(np.int32)
    vf["item_id"] = vf["item_index"]
    vf.drop(columns=["video_id"], inplace=True)
    vf.drop_duplicates(subset=["item_index"], keep="last", inplace=True)

    item_vs = {}
    for col in ITEM_STATIC_FEATURES:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
        vocab = build_ordered_vocab(vf[col], start=1)
        vf[col] = vf[col].map(vocab).astype(np.int32)
        item_vs[col] = len(vocab) + 1

    num_items = max(item_idx_map.values())
    item_info = pd.DataFrame(
        {
            "item_index": np.arange(num_items + 1, dtype=np.int32),
            "item_id": np.zeros(num_items + 1, dtype=np.int32),
            **{c: np.zeros(num_items + 1, dtype=np.int32) for c in ITEM_STATIC_FEATURES},
        }
    )

    vf_idx = vf.set_index("item_index")
    mask = item_info["item_index"].isin(vf_idx.index)
    matched = item_info.loc[mask, "item_index"].values
    for col in ["item_id"] + ITEM_STATIC_FEATURES:
        item_info.loc[mask, col] = vf_idx.loc[matched, col].values
    item_info = item_info.astype(np.int32)

    item_info.to_parquet(
        output_dir / "item_info.parquet", index=False, engine="pyarrow"
    )
    print(f"  [Saved] item_info.parquet ({num_items + 1:,} items incl. padding)")
    del vf, vf_idx
    return item_vs


def build_item_vocabs_from_data(
    data_dir: Path,
    item_idx_map: dict,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """Build item feature vocab to support OOV filtering.

    Args:
        data_dir: data directory
        item_idx_map: {video_id: global_item_index}
        feat_value_counters: {feature_name: {value_str: count}}
        min_feat_count: frequency threshold

    Returns:
        {feature_name: {value_str: int_id}}
    """
    vf = load_video_basic_features(data_dir)
    vf = vf[vf["video_id"].isin(item_idx_map)].copy()

    vocabs = {}
    for col in ITEM_STATIC_FEATURES:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
        vocab = {}
        idx = 1
        all_vals = set(vf[col].unique()) | {"__MISSING__"}
        if col in feat_value_counters:
            counter = feat_value_counters[col]
            for v in sorted(all_vals):
                cnt = int(counter.get(v, 0))
                if cnt > 0 and cnt < min_feat_count:
                    continue
                vocab[v] = idx
                idx += 1
        else:
            for v in sorted(all_vals):
                vocab[v] = idx
                idx += 1
        vocabs[col] = vocab

    return vocabs


# ================================================================
#  7. Main process
# ================================================================

def preprocess_and_split(
    data_dir: str = "./KuaiRand-27K/data",
    output_dir: str = "./data/KuaiRand_Video_Action",
    min_user_interactions: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_user_parts: int = 50,
    chunk_size: int = 4_000_000,
    buffer_flush_size: int = 1_000_000,
    train_blocks: int = 32,
    valid_blocks: int = 8,
    test_blocks: int = 8,
    min_feat_count: int = MIN_FEAT_COUNT,
    num_workers: int = 4,
    overwrite: bool = False,
):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    block_counts = (int(train_blocks), int(valid_blocks), int(test_blocks))
    if min(block_counts) <= 0:
        raise ValueError("train_blocks / valid_blocks / test_blocks must be positive integers.")
    target_blocks = max(block_counts)
    if n_user_parts < target_blocks:
        print(
            f"[Config] n_user_parts={n_user_parts} is less than the maximum target block number {target_blocks},"
            f"Automatically adjust to {target_blocks} to prevent all target blocks from being generated."
        )
        n_user_parts = target_blocks
    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be a positive integer.")

    # Security check: output_dir is prohibited from being equal to data_dir or an ancestor directory of data_dir.
    # Otherwise, overwrite will also delete the original data.
    if data_dir == output_dir or data_dir.is_relative_to(output_dir):
        raise ValueError(
            f"output_dir cannot be equal to or contain data_dir, otherwise the original data will be deleted! \n"
            f"  data_dir:   {data_dir}\n"
            f"  output_dir: {output_dir}\n"
            f"Please set output_dir to a different directory than data_dir."
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Please remove it first, or use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_partitions"

    # ================================================================
    #  Step 1/9: Load user features + build initial vocab (for Phase 1 intermediate coding)
    # ================================================================
    print("\n[Step 1/9] Load user features & build vocab")
    uf = load_user_features(data_dir)
    vocabs_initial = build_all_vocabs(uf)
    pat2name, name2code = build_action_maps()
    uf_enc = encode_user_features_to_int(uf, vocabs_initial)
    print(f"  action type: {len(name2code)}")
    print("  User features have been precoded as int16 (intermediate encoding, OOV filtering will be done later)\n")

    # ================================================================
    #  Step 2/9: Phase 1 — CSV → Partition Parquet
    # ================================================================
    print("[Step 2/9] Phase 1: CSV → Partition Parquet")
    user_counts, item_ids, all_dates, _ = phase1_partition_to_parquet(
        data_dir,
        tmp_dir,
        uf_enc,
        vocabs_initial,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )

    # ================================================================
    #  Step 2.5/9: Calculate the frequency of feature values based on the number of user interactions (for OOV filtering)
    # ================================================================
    print("[Step 2.5/9] Calculate eigenvalue frequency (for OOV filtering)")
    feat_value_counters = compute_feat_value_counters(user_counts, uf_enc)
    for col in USER_STATIC_FEATURES:
        n_total = len(feat_value_counters[col])
        n_kept = sum(1 for v, c in feat_value_counters[col].items() if c >= min_feat_count or v == 0)
        print(f"  {col}: total_values={n_total}, kept_after_oov_filter={n_kept}")

    # ================================================================
    #  Step 3/9: Build the final vocab based on OOV filtering
    # ================================================================
    print("[Step 3/9] Build the final vocab based on OOV filtering")
    vocabs = build_vocabs_with_oov_filter(
        uf, feat_value_counters, min_feat_count=min_feat_count
    )
    del uf, uf_enc, vocabs_initial, feat_value_counters
    gc.collect()
    print(f"  OOV filter threshold (min_feat_count): {min_feat_count}")
    print(f"  action type: {len(name2code)}\n")

    # ================================================================
    #  Step 4/9: Determine the split date according to the proportion of days
    # ================================================================
    print("[Step 4/9] Determine the split date according to the date ratio")
    sorted_dates = sorted(all_dates)
    split_info = build_date_split(
        sorted_dates,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    train_dates = split_info["train_dates"]
    valid_dates = split_info["valid_dates"]
    test_dates = split_info["test_dates"]
    valid_start_date = split_info["valid_start_date"]
    test_start_date = split_info["test_start_date"]

    print(f"  Total days: {split_info['n_days']} days ({fmt_date_int(sorted_dates[0])} ~ {fmt_date_int(sorted_dates[-1])})")
    print(f"  Training set: {split_info['n_train']} days {fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}")
    print(f"  Validation set: {split_info['n_valid']} days {fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}")
    print(f"  Test set: {split_info['n_test']} day {fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}")
    print(f"  Day ratio: {split_info['n_train']}:{split_info['n_valid']}:{split_info['n_test']}\n")

    # ================================================================
    #  Step 5/9: Filter low-frequency users + build global ID mapping + vocab_size
    # ================================================================
    print("[Step 5/9] Filter low-frequency users + build ID mapping")
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    n_dropped = len(user_counts) - len(valid_users)
    print(f"  Valid users: {len(valid_users):,} (filter {n_dropped:,})")

    sorted_users = sorted(valid_users)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}

    sorted_items = sorted(item_ids)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  Number of videos: {len(item_idx_map):,}")

    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "action": len(name2code) + 1,
        "timestamp": 0,
    }
    for col in USER_STATIC_FEATURES + CONTEXT_FEATURES:
        vocab_size[col] = len(vocabs[col]) + 1

    del user_counts, item_ids, sorted_users, sorted_items
    gc.collect()
    print()

    # ================================================================
    #  Step 6/9: Phase 2 — Encoding partition by partition + segmentation by date + writing block data/user_info
    # ================================================================
    print("[Step 6/9] Encode partition by partition + split by date + write block data/user_info")

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }

    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    # Reload uf_enc for Phase 2 (previously deled)
    # However, the encoding of uf_enc is based on the int16 of the initial vocab. In Phase 2, the OOV filtered vocab needs to be applied.
    # So uf_enc still needs to be passed
    uf_phase2 = load_user_features(data_dir)
    uf_enc_phase2 = encode_user_features_to_int(
        uf_phase2, build_all_vocabs(uf_phase2)
    )
    del uf_phase2

    def consume_partition_result(pid, result):
        nonlocal max_seq
        if result is None:
            return

        user_info_rows = result["user_info"]
        if len(user_info_rows) > 0:
            for r in user_info_rows:
                sl = len(r["full_item_seq"])
                if sl > max_seq:
                    max_seq = sl

        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            managers[split_name].append_partition(
                pid=pid,
                split_df=sdf,
                user_info_rows=user_info_rows,
            )
            sample_counts[split_name] += len(sdf)

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: accumulated {done:,} rows")

        gc.collect()

    if num_workers == 1:
        for pid in range(n_user_parts):
            result = process_partition(
                tmp_dir,
                pid,
                valid_users,
                user_idx_map,
                item_idx_map,
                uf_enc_phase2,
                vocabs,
                pat2name,
                name2code,
                valid_start_date,
                test_start_date,
            )
            consume_partition_result(pid, result)
            del result
    else:
        phase2_result_dir = tmp_dir / "_phase2_results"
        phase2_result_dir.mkdir(parents=True, exist_ok=True)
        actual_workers = min(int(num_workers), int(n_user_parts))
        print(f"  Phase 2 multiprocessing: num_workers={actual_workers}")

        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_init_phase2_worker,
            initargs=(
                tmp_dir,
                phase2_result_dir,
                valid_users,
                user_idx_map,
                item_idx_map,
                uf_enc_phase2,
                vocabs,
                pat2name,
                name2code,
                valid_start_date,
                test_start_date,
            ),
        ) as executor:
            pending = {}
            next_submit_pid = 0
            while next_submit_pid < actual_workers:
                pending[next_submit_pid] = executor.submit(
                    _process_partition_to_temp, next_submit_pid
                )
                next_submit_pid += 1

            for pid in range(n_user_parts):
                payload = pending.pop(pid).result()
                if payload["pid"] != pid:
                    raise RuntimeError(
                        f"Phase 2 worker returned pid={payload['pid']}, expected pid={pid}."
                    )
                if next_submit_pid < n_user_parts:
                    pending[next_submit_pid] = executor.submit(
                        _process_partition_to_temp, next_submit_pid
                    )
                    next_submit_pid += 1

                if payload["empty"]:
                    continue

                user_info_df = pd.read_parquet(payload["user_info_file"])
                user_info_rows = user_info_df.to_dict(orient="records")
                result = {"user_info": user_info_rows}
                for split_name in ["train", "valid", "test"]:
                    split_file = payload["split_files"].get(split_name)
                    if split_file is None:
                        result[split_name] = pd.DataFrame(columns=FINAL_COLUMNS)
                    else:
                        result[split_name] = pd.read_parquet(split_file)

                consume_partition_result(pid, result)
                shutil.rmtree(payload["result_dir"], ignore_errors=True)
                del result, user_info_df, user_info_rows, payload
                gc.collect()

    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  Total={total:,}\n")

    if total == 0:
        raise ValueError(
            "The processed data is empty, please check the original data or reduce min_user_interactions."
        )

    del valid_users, user_idx_map, uf_enc_phase2
    gc.collect()

    # ================================================================
    #  Step 7/9: Build item_info for each block
    # ================================================================
    print("\n[Step 7/9] Build item_info for each block")

    # Load the original video feature table (for subsequent item vocab and global_item_lookup)
    item_feat_raw = load_video_basic_features(data_dir)

    # Build item feature vocab
    # To simplify the implementation, OOV filtering is not performed on the item feature (only OOV filtering is performed on the user feature).
    # Because the frequency statistics of item features require scanning the number of occurrences of each item in the log,
    # The item feature value is loaded from an independent file and is independent of the log frequency.
    item_vs = {}
    for col in ITEM_STATIC_FEATURES:
        item_feat_raw[col] = item_feat_raw[col].astype(str).replace(
            {"nan": "__MISSING__", "": "__MISSING__"}
        )
        vocab = build_ordered_vocab(item_feat_raw[col], start=1)
        item_feat_raw[col] = item_feat_raw[col].map(vocab).astype(np.int32)
        item_vs[col] = len(vocab) + 1

    global_item_lookup = build_global_item_lookup(item_feat_raw, item_idx_map)

    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    del item_feat_raw, global_item_lookup
    gc.collect()

    # ================================================================
    #  Step 8/9: Save meta_data + block_manifest
    # ================================================================
    print("\n[Step 8/9] Save meta_data + block_manifest")

    full_vs = dict(vocab_size)
    full_vs.update(item_vs)

    meta = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
        },
        "split_by_date": {
            "train_days": split_info["n_train"],
            "train_range": f"{fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}",
            "valid_days": split_info["n_valid"],
            "valid_range": f"{fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}",
            "test_days": split_info["n_test"],
            "test_range": f"{fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}",
        },
        "oov_filter": {
            "min_feat_count": int(min_feat_count),
            "applied_to": "USER_STATIC_FEATURES (based on the number of user occurrences in the log)",
            "rule": "The uniform mapping of feature value occurrences < min_feat_count is 0 (unknown/padding)",
        },
        "blocked_layout": {
            "n_user_parts": int(n_user_parts),
            "num_workers": int(num_workers),
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "train": {
                "data_dir": "train/data",
                "user_info_dir": "train/user_info",
                "item_info_dir": "train/item_info",
            },
            "valid": {
                "data_dir": "valid/data",
                "user_info_dir": "valid/user_info",
                "item_info_dir": "valid/item_info",
            },
            "test": {
                "data_dir": "test/data",
                "user_info_dir": "test/user_info",
                "item_info_dir": "test/item_info",
            },
            "block_pair_rule": (
                "Under the same split, data/user_info/item_info uses the same part-xxxxx number for paired reading."
            ),
            "local_index_rule": {
                "user_index": "block-local dense index, starts from 0",
                "item_index": "block-local dense index, 0 reserved for padding",
                "user_id": "global feature id, consistent across blocks",
                "item_id": "global feature id, consistent across blocks",
            },
        },
        "vocab_size": {k: int(v) for k, v in full_vs.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in name2code.items()},
        "action_vocab_desc": (
            "Encoded action vocabulary for dataloader based on full_action_seq"
            "Construct task-specific token masks."
        ),
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
                "full_timestamp_seq",
            ],
            "full_timestamp_seq_desc": "Raw time_ms sequence in chronological order",
            "desc": (
                "The item index in user_index / full_item_seq here is all block-local index;"
                "full_action_seq / full_timestamp_seq is the global time sequence sequence."
            ),
        },
        "item_info_schema": {
            "fields": [
                "item_index",
                "item_id",
            ] + ITEM_STATIC_FEATURES,
            "desc": (
                "item_index is block-local index; item_id is global item feature id,"
                "Used for embedding consistency."
            ),
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
            "full_timestamp_seq": int(max_seq),
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("  [Saved] meta_data.json")

    block_manifest = {
        "train": managers["train"].buildmanifest(),
        "valid": managers["valid"].buildmanifest(),
        "test": managers["test"].buildmanifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)
    print("  [Saved] block_manifest.json")

    # ================================================================
    #  Step 9/9: Clean up temporary files
    # ================================================================
    print("\n[Step 9/9] Clean up temporary partition files")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Done.")

    # ================================================================
    #  Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  KuaiRand-27K Preprocess Done (split by date ratio + blocked output + OOV filtering)")
    print("=" * 70)
    print(f"Output directory: {output_dir}\n")
    print("Directory structure example:")
    print(f"  {output_dir / 'train' / 'data'}")
    print(f"  {output_dir / 'train' / 'user_info'}")
    print(f"  {output_dir / 'train' / 'item_info'}")
    print(f"  {output_dir / 'valid' / 'data'}")
    print(f"  {output_dir / 'test' / 'data'}")
    print(f"  {output_dir / 'meta_data.json'}")
    print(f"  {output_dir / 'block_manifest.json'}")
    print("\nmeta_data.json:")
    print(json.dumps(meta, ensure_ascii=False, indent=4))


# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess KuaiRand-27K seq-action data into blocked train/valid/test with OOV filtering."
    )
    parser.add_argument("--data_dir", type=str, default="./KuaiRand-27K/data")
    parser.add_argument("--output_dir", type=str, default="../KuaiRand_Video_Action")
    parser.add_argument("--min_user_interactions", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_user_parts", type=int, default=8,
                        help="Phase1 Number of temporary partitions by user hash")
    parser.add_argument("--chunk_size", type=int, default=4_000_000,
                        help="How many rows to process at a time when reading CSV")
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000,
                        help="How much interaction is cached in the temporary partition and then dropped to disk?")
    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT,
                        help="OOV filtering threshold: The number of occurrences of the feature value < the unified mapping of this value is 0")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Phase 2 User partition number of parallel processes; set to 1 to use serial processing")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="If the output directory already exists, delete it and recreate it.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_and_split(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        n_user_parts=args.n_user_parts,
        chunk_size=args.chunk_size,
        buffer_flush_size=args.buffer_flush_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        min_feat_count=args.min_feat_count,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
