"""
数据处理与格式转换工具。

主要功能：
- 从 reward_converted.csv（支持多个 env_id）加载数据
- 自动从 w_* 生成对应的 r_* 列（如果 r_* 不存在或为空）
- 将数据整理成因果发现用的矩阵形式
- 提供生成 LimiX 所需的数表与元数据的接口

说明：
- 数据中只有 w_*, active_*, score 有值，r_* 从 w_* 自动生成
- 训练时使用 r_*, active_*, score（w_* 不进入训练）
- 最终因果图只保存 r_* 和 score（active_* 只用于训练，不进入最终图）
- `env_id` 和 `seed` 作为任务/实验标签提供给 LimiX，但不进入因果图本身
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


VariableType = Literal["continuous", "binary"]


@dataclass
class VariableInfo:
    """单个变量的元信息。"""

    name: str
    var_type: VariableType = "continuous"
    group: Optional[str] = None  # 例如 "reward", "weight", "active"


def load_reward_csv(path: str | Path) -> pd.DataFrame:
    """
    加载 reward_converted.csv，并自动从 w_* 生成对应的 r_* 列。

    规则：
    - 如果某个 w_xxx 存在，但对应的 r_xxx 不存在或全为空，则用 w_xxx 的值填充 r_xxx
    - 这样即使原始数据中 r_* 列都是空的，也能自动生成用于因果图的 r_* 节点

    参数
    ----
    path:
        CSV 文件路径。
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "score" not in df.columns:
        raise ValueError("输入 CSV 必须包含 'score' 列。")
    if "env_id" not in df.columns:
        raise ValueError("输入 CSV 必须包含 'env_id' 列。")

    # 自动从 w_* 生成 r_* 列
    w_cols = [c for c in df.columns if c.startswith("w_")]
    for w_col in w_cols:
        # w_xxx -> r_xxx
        r_col = "r_" + w_col[2:]
        if r_col not in df.columns:
            # 如果 r_* 列不存在，直接创建
            df[r_col] = df[w_col]
        else:
            # 如果 r_* 列存在但全为空/NaN，用 w_* 的值填充
            r_values = df[r_col]
            if r_values.isna().all() or (r_values == 0).all():
                df[r_col] = df[w_col]

    return df


def infer_variable_info(df: pd.DataFrame) -> Dict[str, VariableInfo]:
    """
    根据列名自动推断变量类型与分组。

    规则：
    - 以 'r_' 开头：奖励项，分组 'reward'（进入因果图）
    - 以 'w_' 开头：权重项，分组 'weight'（不进入因果图，只用于生成 r_*）
    - 以 'active_' 开头：是否启用，视为二值，分组 'active'（用于训练，但不进入最终因果图）
    - 'score'：连续变量，分组 'score'（进入因果图）
    其余列忽略（例如 env_id, seed）。
    
    注意：active_* 变量会在训练时使用（帮助学习），但在最终保存和可视化时会被过滤掉。
    """
    info: Dict[str, VariableInfo] = {}
    for col in df.columns:
        if col == "score":
            info[col] = VariableInfo(name=col, var_type="continuous", group="score")
        elif col.startswith("r_"):
            info[col] = VariableInfo(name=col, var_type="continuous", group="reward")
        # w_* 不进入因果图，只作为生成 r_* 的观测来源
        # elif col.startswith("w_"):
        #     info[col] = VariableInfo(name=col, var_type="continuous", group="weight")
        elif col.startswith("active_"):
            # active 列在数据中表现为 0/1
            info[col] = VariableInfo(name=col, var_type="binary", group="active")
        # 其余列（env_id, seed, w_* 等）不进入因果图
    return info


def build_feature_matrix(
    df: pd.DataFrame, var_info: Dict[str, VariableInfo]
) -> pd.DataFrame:
    """
    根据变量信息抽取因果图中的特征列。

    返回
    ----
    new_df:
        仅包含因果图变量（score 与各 r_*, active_*）。
        注意：w_* 不包含在内。
    """
    cols = [name for name in var_info.keys()]
    new_df = df[cols].copy()
    # 简单缺失值处理：用 0 填充；如有更精细需求可自行改写
    new_df = new_df.fillna(0.0)
    return new_df


def build_task_labels(df: pd.DataFrame) -> pd.Series:
    """
    从原始数据中抽取任务标签（env_id）。
    """
    if "env_id" not in df.columns:
        raise ValueError("数据中缺少 'env_id' 列。")
    return df["env_id"].astype(str)


def export_limix_tables(
    df_features: pd.DataFrame,
    task_labels: pd.Series,
    var_info: Dict[str, VariableInfo],
    out_dir: str | Path,
) -> None:
    """
    将特征表与任务标签导出为 LimiX 可读取的通用格式。

    这里不直接依赖 LimiX 的具体 API，而是输出：
    - data.csv：数值表（行：样本，列：变量）
    - tasks.csv：任务/环境标签表
    - meta_variables.csv：变量元信息（类型/分组）

    之后你可以在调用 LimiX 时自行读取这些文件，构造其要求的对象。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_features.to_csv(out_path / "data.csv", index=False)

    pd.DataFrame({"env_id": task_labels}).to_csv(
        out_path / "tasks.csv", index=False
    )

    meta_rows = []
    for v in var_info.values():
        meta_rows.append(
            {
                "name": v.name,
                "type": v.var_type,
                "group": v.group if v.group is not None else "",
            }
        )
    pd.DataFrame(meta_rows).to_csv(out_path / "meta_variables.csv", index=False)


def dataframe_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    简单助手：将特征 DataFrame 转成 numpy 数组，供 DAGMA-MLP 使用。
    """
    return df.to_numpy(dtype=np.float32)


__all__ = [
    "VariableInfo",
    "load_reward_csv",
    "infer_variable_info",
    "build_feature_matrix",
    "build_task_labels",
    "export_limix_tables",
    "dataframe_to_numpy",
]
