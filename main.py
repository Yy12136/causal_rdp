"""
主入口脚本：LimiX + DAGMA-MLP 因果结构学习

流程：
1. 读取 reward_converted.csv（可包含多个 env_id / 任务）
2. 从 w_* 自动生成 r_*（如果不存在）
3. 转换为 LimiX 输入格式（data.csv / tasks.csv / meta_variables.csv）
4. 调用 LimiX 官方模型，基于数据学习 soft prior
5. 整合 yaml 硬约束 + LimiX soft prior，传给 DAGMA-MLP
6. 使用 DAGMA-MLP 学习全局因果图（一次训练）

输出：
- A_dagma_global.npy: 全局邻接矩阵
- variables_global.txt: 全局图变量顺序
- edges_global.csv: 全局图边表（含权重）
- graph_global.png: 全局因果图可视化
（不再输出各环境的 A_dagma_env_*.npy、variables_env_*.txt、graph_env_*.png）
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd

# 设置 CUDA 库路径（确保 PyTorch 能找到 CUDA 库）
cuda_lib_paths = [
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12/lib64",
    "/usr/local/cuda-12.2/lib64",
]
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
new_ld = current_ld
for path in cuda_lib_paths:
    if os.path.exists(path) and path not in new_ld:
        new_ld = f"{new_ld}:{path}" if new_ld else path
if new_ld != current_ld:
    os.environ["LD_LIBRARY_PATH"] = new_ld

from data_utils import (
    build_feature_matrix,
    build_task_labels,
    dataframe_to_numpy,
    infer_variable_info,
    load_reward_csv,
    export_limix_tables,
)
from dagma_mlp import DagmaHyperParams, DagmaMLP
from limix_interface import run_limix_ldm_placeholder
from visualize_graphs import visualize_all_graphs


def run_global_dagma_with_path(csv_path: Path, project_root: Path) -> tuple[np.ndarray, list[str]]:
    """
    使用全部数据训练一次 DAGMA，得到全局因果图。

    参数
    ----
    csv_path: 输入 CSV 文件路径
    project_root: 项目根目录（用于生成 limix_input 目录）

    返回
    ----
    A_global: 全局邻接矩阵
    var_names: 变量名列表（与 A_global 的行列顺序一致）
    """

    # 1. 加载原始数据
    df_raw = load_reward_csv(csv_path)

    # 2. 推断变量信息 & 构造特征表
    var_info = infer_variable_info(df_raw)
    df_features = build_feature_matrix(df_raw, var_info)
    task_labels = build_task_labels(df_raw)

    # 3. 导出通用 LimiX 输入表
    limix_input_dir = project_root / "limix_input"
    export_limix_tables(df_features, task_labels, var_info, limix_input_dir)

    # 4. 调用 LimiX，获得约束（硬约束来自 yaml，软约束来自 LimiX 学习）
    var_names = list(var_info.keys())
    limix_constraints = run_limix_ldm_placeholder(limix_input_dir, var_names)

    # 5. 使用 DAGMA-MLP 学习完整因果图（一次训练）
    X_np = dataframe_to_numpy(df_features)
    hparams = DagmaHyperParams(batch_size=64)
    dagma = DagmaMLP(d=len(var_names), limix=limix_constraints, hparams=hparams)
    A_learned = dagma.fit(X_np)

    # 6. 过滤掉 active_* 变量（它们只用于训练，不进入最终因果图）
    # 保留的变量：score 和所有 r_* 变量
    keep_indices = []
    keep_var_names = []
    for i, var_name in enumerate(var_names):
        if var_name == "score" or var_name.startswith("r_"):
            keep_indices.append(i)
            keep_var_names.append(var_name)
    
    # 从邻接矩阵中提取对应的行和列
    A_filtered = A_learned[np.ix_(keep_indices, keep_indices)]
    
    return A_filtered, keep_var_names


def main():
    project_root = Path(__file__).resolve().parent
    
    # 打印 GPU 信息
    import torch
    print("=" * 60)
    print("GPU 检测")
    print("=" * 60)
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  使用 CPU 模式（速度较慢）")
    print()
    
    # ========== 配置路径（可以修改这里） ==========
    # 输入文件路径
    input_csv = project_root / "data" / "reward_converted_merged.csv"  # 修改这里使用合并后的文件
    # 如果合并后的文件在根目录，可以改为：
    # input_csv = project_root / "reward_converted_merged.csv"
    
    # 输出目录（结果文件保存位置）
    output_dir = project_root / "output"  # 修改这里可以改变输出目录
    # 先清空再生成：删除已有内容，避免追加旧文件
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 边阈值：只保留权重大于等于此阈值的边（用于过滤全局图中的弱边）
    # 可以根据需要调整，例如：0.01, 0.05, 0.1 等
    # 值越大，保留的边越少，只显示最重要的因果关系
    edge_threshold = 0.00015 # 修改这里可以调整阈值
    # ============================================

    # 1. 训练全局 DAG
    print("=" * 60)
    print("步骤 1: 训练全局因果图（使用全部数据）")
    print("=" * 60)
    print(f"输入文件: {input_csv}")
    print("\n说明：")
    print("  - 全局图使用所有环境（env_id）的数据一起训练")
    print("  - 得到包含所有变量之间因果关系的完整图")
    print("  - 这个图反映了所有环境中的通用因果结构")
    A_global, var_names = run_global_dagma_with_path(input_csv, project_root)

    # 应用阈值过滤：只保留权重大于等于阈值的边
    print(f"\n应用边阈值过滤（阈值 = {edge_threshold}）...")
    edge_count_before = np.sum(np.abs(A_global) > 1e-6)
    A_global_filtered = A_global.copy()
    A_global_filtered[np.abs(A_global_filtered) < edge_threshold] = 0.0
    edge_count_after = np.sum(np.abs(A_global_filtered) > 1e-6)
    print(f"  过滤前边数: {edge_count_before}")
    print(f"  过滤后边数: {edge_count_after}")
    print(f"  过滤掉边数: {edge_count_before - edge_count_after}")
    
    # 保存过滤后的全局图
    out_path_A = output_dir / "A_dagma_global.npy"
    out_path_vars = output_dir / "variables_global.txt"
    np.save(out_path_A, A_global_filtered)
    with open(out_path_vars, "w", encoding="utf-8") as f:
        f.write("\n".join(var_names))
    print(f"\n已保存全局邻接矩阵: {out_path_A}")
    print(f"已保存变量顺序: {out_path_vars}")
    
    # 生成边的CSV表格（包含权重值）
    edges_list = []
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            weight = A_global_filtered[i, j]
            if abs(weight) >= edge_threshold:  # 只包含大于等于阈值的边
                # 去掉变量名中的 "r_" 前缀（如果存在）
                source_name = var_names[i]
                target_name = var_names[j]
                if source_name.startswith("r_"):
                    source_name = source_name[2:]  # 去掉 "r_" 前缀
                if target_name.startswith("r_"):
                    target_name = target_name[2:]  # 去掉 "r_" 前缀
                
                edges_list.append({
                    "source": source_name,
                    "target": target_name,
                    "weight": weight
                })
    
    if edges_list:
        edges_df = pd.DataFrame(edges_list)
        # 按权重绝对值降序排序（最重要的边在前）
        edges_df = edges_df.reindex(edges_df["weight"].abs().sort_values(ascending=False).index)
        edges_df = edges_df.reset_index(drop=True)
        
        out_path_edges = output_dir / "edges_global.csv"
        edges_df.to_csv(out_path_edges, index=False, encoding="utf-8")
        print(f"已保存边的CSV表格: {out_path_edges}")
        print(f"  包含 {len(edges_df)} 条边")
        print(f"  权重范围: [{edges_df['weight'].min():.6f}, {edges_df['weight'].max():.6f}]")
    else:
        print("⚠️  警告: 没有符合条件的边（可能阈值设置过高）")

    # 2. 仅可视化全局图（不生成各环境的 png）
    print("\n" + "=" * 60)
    print("步骤 2: 可视化全局因果图")
    print("=" * 60)
    visualize_all_graphs(output_dir, edge_threshold=edge_threshold)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"全局图: {out_path_A}")
    print(f"变量顺序: {out_path_vars}")
    print(f"边表格: {output_dir / 'edges_global.csv'}")
    print(f"可视化: {output_dir / 'graph_global.png'}")


if __name__ == "__main__":
    main()
