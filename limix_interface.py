"""
与 LimiX-ldm 的对接接口（逻辑因果先验）。

核心逻辑：
- yaml 只提供硬约束（blacklist/whitelist）："一定不会出现的边" / "一定会存在的边"
- LimiX 官方模型基于数据学习 soft prior（edge_pref）：
  - r_* -> score 的 soft prior：通过 LimiX 回归 score 得到特征重要性
  - r_i -> r_j 的 soft prior：通过 r_* 之间的相关性得到
- DAGMA-MLP 只使用 LimiX 学出来的 soft prior，不再使用 yaml 中的 soft_edges
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
import pandas as pd
import torch
import sys


Edge = Tuple[str, str]  # (parent, child)


@dataclass
class LimixConstraints:
    """
    LimiX 输出/传递给 DAGMA 的结构性信息。

    - A_candidate: 候选解邻接矩阵（形状 [d, d]，按变量顺序）
    - blacklist: 不能出现的有向边集合（来自 yaml 硬约束）
    - whitelist: 必须出现的边集合（来自 yaml 硬约束）
    - edge_pref: 边偏好权重 α_ij，形状同 A_candidate（完全由 LimiX 学习）
    - confidence: LimiX 置信度矩阵 C_LimiX，形状 [d,d]，取值 [0,1]；用于加权损失
                  L_total = L_Data + λ||W⊙(1-C_LimiX)||_1，缓解大模型幻觉
    - groups: 组/层级稀疏信息，每个元素是一组边的索引列表
              （例如同一 reward 族，或 r_i -> r_j 这种模式）
    """

    var_names: List[str]
    A_candidate: np.ndarray
    blacklist: List[Edge]
    whitelist: List[Edge]
    edge_pref: np.ndarray
    confidence: np.ndarray  # C_LimiX，由 edge_pref 归一化到 [0,1] 得到
    groups: List[List[Tuple[int, int]]]


def build_default_hard_constraints(var_names: List[str]) -> List[Edge]:
    """
    根据列名自动生成"必须是 r_* -> score"这一类硬约束。

    - score 没有指向其他变量的出边
    - 允许 r_* -> score
    - 默认禁止 score -> 任何 r_*
    """
    blacklist: List[Edge] = []
    if "score" not in var_names:
        return blacklist

    for name in var_names:
        if name != "score":
            blacklist.append(("score", name))  # 禁止 score 作为父结点
    return blacklist


def _try_build_edge_pref_with_limix(
    data_csv_dir: Path,
    var_names: List[str],
    edge_pref: np.ndarray,
) -> None:
    """
    使用本地 LimiX-2M 模型，基于数据学习 soft prior（edge_pref）。

    学习内容：
    1. r_* -> score 的 soft prior：
       - 用 LimiX-2M 对 score 做回归
       - 计算每个 r_* 特征与预测 score 的相关性
       - 重要性越大，对应 r_* -> score 的惩罚越小

    2. r_i -> r_j 的 soft prior：
       - 计算数据中 r_* 之间的相关性
       - 相关性越大，对应 r_i -> r_j 的惩罚越小
       - 不限制方向，DAGMA 可以自由学习 r_i -> r_j 或 r_j -> r_i

    注意：
    - 如果任何一步失败（例如 LimiX 未安装、模型文件缺失），将静默退回，不抛异常。
    - 不会覆盖已有的 edge_pref（例如来自其他来源），而是在其基础上叠加。
    """
    try:
        # 1. 准备路径：本地 LimiX 仓库 + 模型 + 配置
        root_dir = Path("/workspace/causal_rdp/LimiX").resolve()
        model_path = root_dir / "cache" / "LimiX-2M.ckpt"
        
        # 根据设备选择配置文件：CPU 不支持 retrieval，需要使用 noretrieval
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            config_path = root_dir / "config" / "reg_default_noretrieval.json"
            print("\n[LimiX] 检测到 CPU 设备，使用 noretrieval 配置")
        else:
            config_path = root_dir / "config" / "reg_default_2M_retrieval.json"
            print("\n[LimiX] 检测到 GPU 设备，使用 retrieval 配置")

        print("[LimiX] 开始加载模型...")
        print(f"  模型路径: {model_path}")
        print(f"  配置路径: {config_path}")
        
        if not model_path.exists():
            print(f"  ❌ 模型文件不存在: {model_path}")
            return
        if not config_path.exists():
            print(f"  ❌ 配置文件不存在: {config_path}")
            return
        
        print(f"  ✅ 模型文件存在，大小: {model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  ✅ 配置文件存在")

        # 2. 读取 data.csv
        data_path = Path(data_csv_dir) / "data.csv"
        if not data_path.exists():
            print(f"  ❌ data.csv 文件不存在: {data_path}")
            return

        print(f"  [LimiX] 读取数据文件: {data_path}")
        df = pd.read_csv(data_path)
        if "score" not in df.columns:
            print(f"  ❌ data.csv 中没有 score 列")
            return

        # 对齐顺序：按 var_names 重新排序列（安全起见）
        cols_in_df = [c for c in var_names if c in df.columns]
        df = df[cols_in_df].copy()

        if "score" not in df.columns:
            print(f"  ❌ 对齐后没有 score 列")
            return

        print(f"  ✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")

        # 3. 构造特征与标签
        y = df["score"].to_numpy(dtype=np.float32)
        feature_cols = [c for c in df.columns if c != "score"]
        if not feature_cols:
            print(f"  ❌ 没有特征列（除了 score）")
            return
        X = df[feature_cols].to_numpy(dtype=np.float32)
        print(f"  ✅ 特征矩阵: {X.shape[0]} 样本, {X.shape[1]} 特征")

        # 4. 导入 LimiX 的 LimiXPredictor（离线模式，只用本地 ckpt）
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        try:
            from inference.predictor import LimiXPredictor  # type: ignore
        except ImportError as e:
            print(f"  ❌ 无法导入 LimiXPredictor: {e}")
            print(f"  💡 提示: 需要安装 kditransform 依赖")
            print(f"     运行: pip install kditransform")
            print(f"     或者安装完整依赖: pip install kditransform hyperopt")
            print(f"  ⚠️  将跳过 LimiX 学习，只使用硬约束")
            return
        except Exception as e:
            print(f"  ❌ 无法导入 LimiXPredictor: {e}")
            print(f"  ⚠️  将跳过 LimiX 学习，只使用硬约束")
            return

        # device 已经在上面定义了，这里只是打印
        print(f"  使用设备: {device}")

        print("  [LimiX] 正在初始化预测器...")
        predictor = LimiXPredictor(
            device=device,
            model_path=str(model_path),
            inference_config=str(config_path),
            mask_prediction=False,
            inference_with_DDP=False,
        )
        print("  ✅ LimiX 模型加载成功！")

        # 5. 使用 LimiX 做一次 score 回归
        #    简化处理：用全部数据同时作为 train/test，只为得到 y_hat。
        print(f"  [LimiX] 开始回归预测 (样本数: {X.shape[0]}, 特征数: {X.shape[1]})...")
        y_pred = predictor.predict(X, y, X, task_type="Regression")
        print("  ✅ LimiX 回归预测完成")

        # LimiX 回归输出通常是 torch.Tensor，形状 [n_samples, 1] 或 [n_samples]
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.asarray(y_pred).reshape(-1)

        if y_pred.shape[0] != X.shape[0]:
            return

        # 6. 计算每个特征与预测 score 的相关性，作为重要性
        print("  [LimiX] 计算特征重要性...")
        n_features = X.shape[1]
        importance = np.zeros(n_features, dtype=np.float32)
        for j in range(n_features):
            xj = X[:, j]
            if np.allclose(xj, xj[0]):
                importance[j] = 0.0
                continue
            corr = np.corrcoef(xj, y_pred)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            importance[j] = abs(float(corr))

        max_imp = float(importance.max())
        if max_imp <= 0.0:
            print("  ⚠️  所有特征重要性为0，跳过")
            return

        importance = importance / (max_imp + 1e-8)

        # 打印 Top 5 重要特征
        importance_list = [(feat, imp) for feat, imp in zip(feature_cols, importance)]
        importance_list.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 5 重要特征:")
        for feat, imp in importance_list[:5]:
            print(f"    {feat}: {imp:.4f}")

        # 7. 将重要性映射到 edge_pref：r_* -> score 边
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        if "score" not in name_to_idx:
            return
        score_idx = name_to_idx["score"]

        r_to_score_count = 0
        for feat_name, imp in zip(feature_cols, importance):
            if not feat_name.startswith("r_"):
                continue
            if feat_name not in name_to_idx:
                continue
            i = name_to_idx[feat_name]
            # 惩罚权重：1 - importance，重要性越大，惩罚越小
            weight = 1.0 - float(imp)
            # 在原有 edge_pref 基础上叠加
            edge_pref[i, score_idx] += weight
            r_to_score_count += 1
        
        print(f"  ✅ 已学习 {r_to_score_count} 条 r_* -> score 的 soft prior")

        # 8. 学习 r_i -> r_j 的 soft prior：基于 r_* 之间的相关性
        print("  [LimiX] 计算 r_* 之间的相关性...")
        r_names = [name for name in var_names if name.startswith("r_")]
        if len(r_names) >= 2:
            r_df = df[r_names].copy()
            r_mat = r_df.to_numpy(dtype=np.float32)
            # 计算相关性矩阵
            corr_mat = np.corrcoef(r_mat, rowvar=False)
            corr_mat = np.abs(corr_mat)  # 只关心相关性强度，不关心正负
            np.fill_diagonal(corr_mat, 0.0)  # 自己到自己的相关性设为 0

            max_corr = float(corr_mat.max())
            if max_corr > 0.0:
                corr_norm = corr_mat / (max_corr + 1e-8)
                
                # 对每一对 r_i, r_j，给两个方向都加上 soft prior
                # （不限制方向，让 DAGMA 自己决定）
                r_to_r_count = 0
                for a, ra in enumerate(r_names):
                    for b, rb in enumerate(r_names):
                        if a == b:
                            continue
                        imp_ij = float(corr_norm[a, b])
                        if imp_ij > 0.1:  # 只记录相关性较强的
                            w_ij = 1.0 - imp_ij  # 相关性越大，惩罚越小
                            ia = name_to_idx.get(ra)
                            jb = name_to_idx.get(rb)
                            if ia is None or jb is None:
                                continue
                            # 对两个方向都加上 soft prior（让 DAGMA 自己选择方向）
                            edge_pref[ia, jb] += w_ij
                            r_to_r_count += 1
                            # 注意：这里也可以只加一个方向，看你的需求
                            # 如果只想要单向，可以注释掉下面这行
                            # edge_pref[jb, ia] += w_ij
                
                print(f"  ✅ 已学习 {r_to_r_count} 条 r_i -> r_j 的 soft prior (相关性 > 0.1)")
            else:
                print("  ⚠️  r_* 之间没有相关性，跳过")
        else:
            print(f"  ⚠️  r_* 变量数量不足 ({len(r_names)} < 2)，跳过 r_i -> r_j 学习")
        
        print("[LimiX] ✅ 学习完成！\n")

    except Exception as e:
        # 打印错误信息，方便调试
        print(f"[LimiX] ❌ 学习过程出错: {e}")
        import traceback
        traceback.print_exc()
        return


def run_limix_ldm_placeholder(
    data_csv_dir: str | Path,
    var_names: List[str],
) -> LimixConstraints:
    """
    整合 LimiX 模型输出和 yaml 硬约束，生成传给 DAGMA 的约束集合。

    逻辑：
    1. 硬约束（blacklist/whitelist）：完全来自 limix_config.yaml
    2. 软约束（edge_pref）：完全由 LimiX 官方模型基于数据学习
    3. yaml 中的 soft_edges 不再使用（只作为注释保留）

    返回
    ----
    LimixConstraints:
        包含候选结构、硬约束、软约束的完整约束集合
    """
    d = len(var_names)
    A_candidate = np.zeros((d, d), dtype=np.float32)
    edge_pref = np.zeros_like(A_candidate)

    # 1. 默认硬约束：score 不能指向其他变量
    blacklist: List[Edge] = build_default_hard_constraints(var_names)
    whitelist: List[Edge] = []

    # 2. 读取 limix_config.yaml 中的硬约束（blacklist/whitelist）
    print("=" * 60)
    print("步骤 2: 读取硬约束配置")
    print("=" * 60)
    config_path = Path(data_csv_dir).parent / "limix_config.yaml"
    if config_path.exists():
        print(f"读取配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # 2.1 解析硬约束（黑名单 / 白名单）
        hard = (cfg.get("hard_edges") or {})
        for item in hard.get("blacklist", []) or []:
            try:
                u, v = [s.strip() for s in item.split("->")]
                blacklist.append((u, v))
            except Exception:
                # 格式错误时忽略该条
                pass

        for item in hard.get("whitelist", []) or []:
            try:
                # 支持两种格式：
                # 1. 字符串格式: "r_pos_reward -> score"
                # 2. 字典格式: {edge: "r_pos_reward -> score", alpha: 1.0}
                if isinstance(item, str):
                    u, v = [s.strip() for s in item.split("->")]
                    whitelist.append((u, v))
                elif isinstance(item, dict):
                    edge_str = item.get("edge", "")
                    if edge_str:
                        u, v = [s.strip() for s in edge_str.split("->")]
                        whitelist.append((u, v))
            except Exception:
                pass

        print(f"  黑名单 (blacklist): {len(blacklist)} 条")
        if blacklist:
            for u, v in blacklist[:5]:  # 只显示前5条
                print(f"    {u} -> {v}")
            if len(blacklist) > 5:
                print(f"    ... 还有 {len(blacklist) - 5} 条")
        
        print(f"  白名单 (whitelist): {len(whitelist)} 条")
        if whitelist:
            for u, v in whitelist[:5]:  # 只显示前5条
                print(f"    {u} -> {v}")
            if len(whitelist) > 5:
                print(f"    ... 还有 {len(whitelist) - 5} 条")
        
        # 2.2 yaml 中的 soft_edges 现在只作为"可行域提示"，
        #     不再直接转成 edge_pref，避免人工软约束主导学习；
        #     具体 soft prior 交由下方 LimiX 基于数据自动生成。
        # （注释掉原来的 soft_edges 解析代码）
    else:
        print(f"⚠️  配置文件不存在: {config_path}，使用默认硬约束")

    # 3. 使用本地 LimiX-2M 模型，基于数据学习 soft prior（edge_pref）
    #    包括：r_* -> score 和 r_i -> r_j 的 soft prior
    print("=" * 60)
    print("步骤 3: LimiX 学习 soft prior")
    print("=" * 60)
    _try_build_edge_pref_with_limix(Path(data_csv_dir), var_names, edge_pref)
    
    # 保存学习到的 edge_pref 矩阵为 CSV
    output_dir = Path(data_csv_dir).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_pref_df = pd.DataFrame(edge_pref, index=var_names, columns=var_names)
    edge_pref_csv = output_dir / "limix_edge_pref_matrix.csv"
    edge_pref_df.to_csv(edge_pref_csv)
    print(f"✅ edge_pref 矩阵已保存到: {edge_pref_csv}")
    
    # 打印 edge_pref 统计信息
    non_zero_count = np.count_nonzero(edge_pref)
    print(f"  非零边数量: {non_zero_count} / {edge_pref.size}")
    max_weight = edge_pref.max()
    min_weight = edge_pref.min()
    if non_zero_count > 0:
        avg_weight = edge_pref[edge_pref > 0].mean()
    else:
        avg_weight = 0.0
    print(f"  最大权重: {max_weight:.4f}, 最小权重: {min_weight:.4f}, 平均权重: {avg_weight:.4f}")

    # 4. 组稀疏：将所有 r_* -> score 视作一组
    groups: List[List[Tuple[int, int]]] = []
    if "score" in var_names:
        score_idx = var_names.index("score")
        group_edges: List[Tuple[int, int]] = []
        for i, name in enumerate(var_names):
            if name.startswith("r_"):
                group_edges.append((i, score_idx))
    if group_edges:
        groups.append(group_edges)

    # 由 edge_pref 得到置信度 C_LimiX ∈ [0,1]，用于加权损失（高 pref → 高置信度）
    ep = edge_pref
    ep_min, ep_max = ep.min(), ep.max()
    if ep_max > ep_min:
        confidence = (ep - ep_min) / (ep_max - ep_min + 1e-8)
    else:
        confidence = np.ones_like(ep) * 0.5  # 无区分时取 0.5

    return LimixConstraints(
        var_names=var_names,
        A_candidate=A_candidate,
        blacklist=blacklist,
        whitelist=whitelist,
        edge_pref=edge_pref,
        confidence=confidence.astype(np.float32),
        groups=groups,
    )


__all__ = ["Edge", "LimixConstraints", "run_limix_ldm_placeholder"]
