"""
与 LimiX-ldm 的对接接口（逻辑因果先验）。

核心逻辑：
- yaml 只提供硬约束（blacklist/whitelist）："一定不会出现的边" / "一定会存在的边"
- The LimiX-based estimator estimates a raw causal-effect score matrix M_raw
  from macro-level reward-component intervention data.
- M_raw 派生出 M_prior 与 M_conf，供 CRWM 联合优化器的 L_soft 使用：
  - r_* -> score：LimiX 回归 score，特征与预测 score 的相关性作为 estimated causal-effect score
  - r_i -> r_j：r_* 之间的相关性作为 estimated causal-effect score
- edge_pref 由 M_raw 派生，保留以兼容旧代码
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
import pandas as pd
import torch
import sys


Edge = Tuple[str, str]  # (parent, child)


@dataclass
class LimixConstraints:
    """
    LimiX 输出/传递给 CRWM 联合优化器的结构性信息。

    - A_candidate: 候选解邻接矩阵（形状 [d, d]，按变量顺序）
    - blacklist: 不能出现的有向边集合（来自 yaml 硬约束）
    - whitelist: 必须出现的边集合（来自 yaml 硬约束）
    - edge_pref: 边惩罚权重，形状 [d, d]（由 M_raw 派生，保留以兼容旧代码）
    - confidence: 由 edge_pref 归一化到 [0,1] 得到（保留，向后兼容）
    - groups: 组/层级稀疏信息
    - M_prior: 先验因果矩阵，[d, d]，等于 M_raw 的拷贝
               对应论文中的 M_prior；由 LimiX estimated causal-effect score 给出
    - M_conf:  M_prior 的置信度，[d, d]，取值 [0, 1]
               对应论文中的 M_conf；用于 L_soft = γ ||M_conf ⊙ (M_inv - M_prior)||_F²
    """

    var_names: List[str]
    A_candidate: np.ndarray
    blacklist: List[Edge]
    whitelist: List[Edge]
    edge_pref: np.ndarray
    confidence: np.ndarray
    groups: List[List[Tuple[int, int]]]
    M_prior: np.ndarray   # 先验因果矩阵，d×d
    M_conf: np.ndarray    # M_prior 置信度，d×d，[0, 1]


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


def _edge_pref_from_m_raw(M_raw: np.ndarray) -> np.ndarray:
    """由 M_raw 派生 edge_pref：非零位置 edge_pref = 1 - normalized_abs_score，零位置为 0。"""
    edge_pref = np.zeros_like(M_raw)
    mp_abs_max = float(np.max(np.abs(M_raw)))
    if mp_abs_max <= 0.0:
        return edge_pref
    normalized_abs = np.abs(M_raw) / (mp_abs_max + 1e-8)
    nonzero = np.abs(M_raw) > 0
    edge_pref[nonzero] = 1.0 - normalized_abs[nonzero]
    return edge_pref


def estimate_limix_raw_effect_matrix(
    data_csv_dir: Path,
    var_names: List[str],
) -> np.ndarray:
    """
    从 macro-level reward-component 数据估计 raw causal-effect score 矩阵 M_raw。

    M_raw[target, source] 表示 source -> target 的 estimated causal-effect score。
    若 LimiX 模型缺失、依赖缺失或预测失败，返回全零矩阵并打印 warning。
    """
    d = len(var_names)
    M_raw = np.zeros((d, d), dtype=np.float32)

    try:
        root_dir = Path("/workspace/causal_rdp/LimiX").resolve()
        model_path = root_dir / "cache" / "LimiX-2M.ckpt"

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
            print(f"[LimiX] ⚠️  Warning: 模型文件不存在: {model_path}，M_raw 返回全零")
            return M_raw
        if not config_path.exists():
            print(f"[LimiX] ⚠️  Warning: 配置文件不存在: {config_path}，M_raw 返回全零")
            return M_raw

        print(f"  ✅ 模型文件存在，大小: {model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  ✅ 配置文件存在")

        data_path = Path(data_csv_dir) / "data.csv"
        if not data_path.exists():
            print(f"[LimiX] ⚠️  Warning: data.csv 不存在: {data_path}，M_raw 返回全零")
            return M_raw

        print(f"  [LimiX] 读取数据文件: {data_path}")
        df = pd.read_csv(data_path)
        if "score" not in df.columns:
            print(f"[LimiX] ⚠️  Warning: data.csv 中没有 score 列，M_raw 返回全零")
            return M_raw

        cols_in_df = [c for c in var_names if c in df.columns]
        df = df[cols_in_df].copy()

        if "score" not in df.columns:
            print(f"[LimiX] ⚠️  Warning: 对齐后没有 score 列，M_raw 返回全零")
            return M_raw

        print(f"  ✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")

        y = df["score"].to_numpy(dtype=np.float32)
        feature_cols = [c for c in df.columns if c != "score"]
        if not feature_cols:
            print(f"[LimiX] ⚠️  Warning: 没有特征列（除了 score），M_raw 返回全零")
            return M_raw
        X = df[feature_cols].to_numpy(dtype=np.float32)
        print(f"  ✅ 特征矩阵: {X.shape[0]} 样本, {X.shape[1]} 特征")

        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        try:
            from inference.predictor import LimiXPredictor  # type: ignore
        except ImportError as e:
            print(f"[LimiX] ⚠️  Warning: 无法导入 LimiXPredictor: {e}")
            print(f"  💡 提示: 需要安装 kditransform 依赖")
            print(f"     运行: pip install kditransform")
            print(f"     或者安装完整依赖: pip install kditransform hyperopt")
            return M_raw
        except Exception as e:
            print(f"[LimiX] ⚠️  Warning: 无法导入 LimiXPredictor: {e}")
            return M_raw

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

        print(f"  [LimiX] 开始回归预测 (样本数: {X.shape[0]}, 特征数: {X.shape[1]})...")
        y_pred = predictor.predict(X, y, X, task_type="Regression")
        print("  ✅ LimiX 回归预测完成")

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.asarray(y_pred).reshape(-1)

        if y_pred.shape[0] != X.shape[0]:
            print("[LimiX] ⚠️  Warning: 预测输出长度与样本数不匹配，M_raw 返回全零")
            return M_raw

        print("  [LimiX] 计算 r_* -> score 的 estimated causal-effect score...")
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
            print("[LimiX] ⚠️  Warning: 所有特征 estimated causal-effect score 为 0")
            return M_raw

        importance = importance / (max_imp + 1e-8)

        importance_list = [(feat, imp) for feat, imp in zip(feature_cols, importance)]
        importance_list.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 5 estimated causal-effect score (r_* -> score):")
        for feat, imp in importance_list[:5]:
            print(f"    {feat}: {imp:.4f}")

        name_to_idx = {name: i for i, name in enumerate(var_names)}
        if "score" not in name_to_idx:
            print("[LimiX] ⚠️  Warning: var_names 中没有 score，M_raw 返回全零")
            return M_raw
        score_idx = name_to_idx["score"]

        r_to_score_count = 0
        for feat_name, imp in zip(feature_cols, importance):
            if not feat_name.startswith("r_"):
                continue
            if feat_name not in name_to_idx:
                continue
            i = name_to_idx[feat_name]
            # 约定 M[target, source]：r_i -> score → M_raw[score_idx, i]
            M_raw[score_idx, i] = float(imp)
            r_to_score_count += 1

        print(f"  ✅ 已估计 {r_to_score_count} 条 r_* -> score 的 causal-effect score")

        print("  [LimiX] 计算 r_i -> r_j 的 estimated causal-effect score...")
        r_names = [name for name in var_names if name.startswith("r_")]
        if len(r_names) >= 2:
            r_df = df[r_names].copy()
            r_mat = r_df.to_numpy(dtype=np.float32)
            corr_mat = np.corrcoef(r_mat, rowvar=False)
            corr_mat = np.abs(corr_mat)
            np.fill_diagonal(corr_mat, 0.0)

            max_corr = float(corr_mat.max())
            if max_corr > 0.0:
                corr_norm = corr_mat / (max_corr + 1e-8)

                r_to_r_count = 0
                for a, ra in enumerate(r_names):
                    for b, rb in enumerate(r_names):
                        if a == b:
                            continue
                        imp_ij = float(corr_norm[a, b])
                        if imp_ij > 0.1:
                            ia = name_to_idx.get(ra)
                            jb = name_to_idx.get(rb)
                            if ia is None or jb is None:
                                continue
                            # 约定 M[target, source]：r_a -> r_b → M_raw[jb, ia]
                            M_raw[jb, ia] = max(M_raw[jb, ia], imp_ij)
                            r_to_r_count += 1

                print(
                    f"  ✅ 已估计 {r_to_r_count} 条 r_i -> r_j 的 causal-effect score "
                    f"(score > 0.1)"
                )
            else:
                print("  ⚠️  r_* 之间没有相关性，跳过 r_i -> r_j 估计")
        else:
            print(f"  ⚠️  r_* 变量数量不足 ({len(r_names)} < 2)，跳过 r_i -> r_j 估计")

        print("[LimiX] ✅ M_raw 估计完成！\n")
        return M_raw

    except Exception as e:
        print(f"[LimiX] ⚠️  Warning: M_raw 估计过程出错: {e}")
        import traceback
        traceback.print_exc()
        return M_raw


def _try_build_edge_pref_with_limix(
    data_csv_dir: Path,
    var_names: List[str],
    edge_pref: np.ndarray,
    m_prior: np.ndarray,
) -> None:
    """
    兼容旧接口：调用 estimate_limix_raw_effect_matrix，将结果写入 edge_pref 与 m_prior。
    """
    M_raw = estimate_limix_raw_effect_matrix(data_csv_dir, var_names)
    m_prior[:] = M_raw
    edge_pref[:] = _edge_pref_from_m_raw(M_raw)


def run_limix_ldm_placeholder(
    data_csv_dir: str | Path,
    var_names: List[str],
) -> LimixConstraints:
    """
    整合 LimiX estimated causal-effect score 与 yaml 硬约束，生成传给 CRWM 的约束集合。

    逻辑：
    1. 硬约束（blacklist/whitelist）：完全来自 limix_config.yaml
    2. M_raw：LimiX-based estimator 从 macro-level 数据估计 raw causal-effect score
    3. M_prior = M_raw；M_conf 由 |M_raw| 归一化；edge_pref 由 M_raw 派生（兼容旧代码）
    4. yaml 中的 soft_edges 不再使用（只作为注释保留）

    返回
    ----
    LimixConstraints:
        包含候选结构、硬约束、M_raw 派生先验的完整约束集合
    """
    d = len(var_names)
    A_candidate = np.zeros((d, d), dtype=np.float32)

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

        hard = (cfg.get("hard_edges") or {})
        for item in hard.get("blacklist", []) or []:
            try:
                u, v = [s.strip() for s in item.split("->")]
                blacklist.append((u, v))
            except Exception:
                pass

        for item in hard.get("whitelist", []) or []:
            try:
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
            for u, v in blacklist[:5]:
                print(f"    {u} -> {v}")
            if len(blacklist) > 5:
                print(f"    ... 还有 {len(blacklist) - 5} 条")

        print(f"  白名单 (whitelist): {len(whitelist)} 条")
        if whitelist:
            for u, v in whitelist[:5]:
                print(f"    {u} -> {v}")
            if len(whitelist) > 5:
                print(f"    ... 还有 {len(whitelist) - 5} 条")
    else:
        print(f"⚠️  配置文件不存在: {config_path}，使用默认硬约束")

    # 3. 估计 M_raw：macro-level reward-component intervention data
    print("=" * 60)
    print("步骤 3: LimiX 估计 M_raw (estimated causal-effect score)")
    print("=" * 60)
    M_raw = estimate_limix_raw_effect_matrix(Path(data_csv_dir), var_names)
    m_prior = M_raw.copy()
    mp_abs_max = float(np.max(np.abs(M_raw)))
    if mp_abs_max > 0.0:
        m_conf = (np.abs(M_raw) / (mp_abs_max + 1e-8)).astype(np.float32)
    else:
        m_conf = np.zeros_like(M_raw)
    edge_pref = _edge_pref_from_m_raw(M_raw)

    # 保存 M_raw / M_prior / M_conf 矩阵
    output_dir = Path(data_csv_dir).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for arr, fname in (
        (M_raw, "limix_M_raw_matrix.csv"),
        (m_prior, "limix_M_prior_matrix.csv"),
        (m_conf, "limix_M_conf_matrix.csv"),
    ):
        pd.DataFrame(arr, index=var_names, columns=var_names).to_csv(
            output_dir / fname
        )
        print(f"✅ {fname} 已保存到: {output_dir / fname}")

    print(f"  M_raw 非零边数: {np.count_nonzero(M_raw)}, max={mp_abs_max:.4f}")
    print(f"  M_prior 非零边数: {np.count_nonzero(m_prior)}")
    print(f"  M_conf 非零边数: {np.count_nonzero(m_conf)}")

    non_zero_ep = np.count_nonzero(edge_pref)
    print(f"  edge_pref 非零边数: {non_zero_ep} / {edge_pref.size} (由 M_raw 派生)")

    # 4. 组稀疏：将所有 r_* -> score 视作一组
    groups: List[List[Tuple[int, int]]] = []
    if "score" in var_names:
        score_idx = var_names.index("score")
        group_edges: List[Tuple[int, int]] = []
        for i, name in enumerate(var_names):
            if name.startswith("r_"):
                group_edges.append((score_idx, i))  # M[target, source]
        if group_edges:
            groups.append(group_edges)

    # 向后兼容：由 edge_pref 得到旧版 confidence
    ep = edge_pref
    ep_min, ep_max = ep.min(), ep.max()
    if ep_max > ep_min:
        confidence = (ep - ep_min) / (ep_max - ep_min + 1e-8)
    else:
        confidence = np.ones_like(ep) * 0.5

    return LimixConstraints(
        var_names=var_names,
        A_candidate=A_candidate,
        blacklist=blacklist,
        whitelist=whitelist,
        edge_pref=edge_pref,
        confidence=confidence.astype(np.float32),
        groups=groups,
        M_prior=m_prior,
        M_conf=m_conf,
    )


__all__ = [
    "Edge",
    "LimixConstraints",
    "estimate_limix_raw_effect_matrix",
    "run_limix_ldm_placeholder",
]
