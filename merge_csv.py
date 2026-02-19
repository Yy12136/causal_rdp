#!/usr/bin/env python3
"""
合并 data 文件夹中的所有 CSV 文件为一个完整的 CSV 文件
"""
import pandas as pd
from pathlib import Path

def merge_csv_files(data_dir: str = "data", output_file: str = "data/reward_converted_merged.csv"):
    """
    合并 data 目录下所有 CSV 文件
    
    Args:
        data_dir: 包含 CSV 文件的目录
        output_file: 输出文件路径
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    # 首先删除旧的合并文件，再重新创建，避免残留或追加
    if output_path.exists():
        output_path.unlink()
        print(f"已删除旧文件: {output_path}")
    
    # 获取所有 CSV 文件（排除合并结果本身，避免把自己读进去）
    csv_files = [f for f in data_path.glob("*.csv") if f.resolve() != output_path.resolve()]
    
    if not csv_files:
        print(f"在 {data_dir} 目录下没有找到 CSV 文件")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # 读取所有 CSV 文件
    dataframes = []
    all_columns = set()
    
    for csv_file in csv_files:
        print(f"\n读取: {csv_file.name}")
        df = pd.read_csv(csv_file)
        print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
        dataframes.append(df)
        all_columns.update(df.columns)
    
    # 统一列顺序：score 在最前，然后是所有 r_*, w_*, active_* 列（按字母排序），最后是 env_id, seed
    score_cols = [c for c in all_columns if c == "score"]
    r_cols = sorted([c for c in all_columns if c.startswith("r_")])
    w_cols = sorted([c for c in all_columns if c.startswith("w_")])
    active_cols = sorted([c for c in all_columns if c.startswith("active_")])
    other_cols = sorted([c for c in all_columns if c not in score_cols + r_cols + w_cols + active_cols])
    
    # 重新排列列：score, 然后按组件分组 (r_, w_, active_)
    # 为了保持一致性，我们需要按组件名分组
    component_groups = {}
    for col in all_columns:
        if col == "score" or col in ["env_id", "seed"]:
            continue
        # 提取组件名（去掉前缀）
        if col.startswith("r_"):
            comp_name = col[2:]
            if comp_name not in component_groups:
                component_groups[comp_name] = {}
            component_groups[comp_name]["r"] = col
        elif col.startswith("w_"):
            comp_name = col[2:]
            if comp_name not in component_groups:
                component_groups[comp_name] = {}
            component_groups[comp_name]["w"] = col
        elif col.startswith("active_"):
            comp_name = col[7:]
            if comp_name not in component_groups:
                component_groups[comp_name] = {}
            component_groups[comp_name]["active"] = col
    
    # 构建统一的列顺序
    unified_columns = ["score"]
    for comp_name in sorted(component_groups.keys()):
        group = component_groups[comp_name]
        if "r" in group:
            unified_columns.append(group["r"])
        if "w" in group:
            unified_columns.append(group["w"])
        if "active" in group:
            unified_columns.append(group["active"])
    
    # 添加其他列（env_id, seed 等）
    for col in ["env_id", "seed"]:
        if col in all_columns:
            unified_columns.append(col)
    
    # 添加其他未分类的列
    for col in other_cols:
        if col not in unified_columns:
            unified_columns.append(col)
    
    print(f"\n统一后的列数: {len(unified_columns)}")
    
    # 为每个 DataFrame 添加缺失的列（填充空值）
    unified_dfs = []
    for i, df in enumerate(dataframes):
        # 找出缺失的列
        missing_cols = [col for col in unified_columns if col not in df.columns]
        
        # 一次性创建所有缺失的列（避免碎片化）
        if missing_cols:
            missing_data = {}
            for col in missing_cols:
                missing_data[col] = ["" if col.startswith("r_") else 0] * len(df)
            missing_df = pd.DataFrame(missing_data, index=df.index)
            # 使用 concat 一次性合并，避免碎片化
            df_unified = pd.concat([df, missing_df], axis=1)
        else:
            df_unified = df.copy()
        
        # 重新排列列顺序（使用 reindex 更高效）
        df_unified = df_unified.reindex(columns=unified_columns)
        unified_dfs.append(df_unified)
        print(f"  文件 {i+1} 处理完成，行数: {len(df_unified)}")
    
    # 合并所有 DataFrame
    print("\n合并所有数据...")
    merged_df = pd.concat(unified_dfs, ignore_index=True)
    
    print(f"合并完成！")
    print(f"  总行数: {len(merged_df)}")
    print(f"  总列数: {len(merged_df.columns)}")
    print(f"  输出文件: {output_path}")
    
    # 保存（合并文件已在函数开头删除，此处直接写入）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    # 显示每个 env_id 的行数统计
    if "env_id" in merged_df.columns:
        print("\n各环境数据统计:")
        env_counts = merged_df["env_id"].value_counts()
        for env_id, count in env_counts.items():
            print(f"  {env_id}: {count} 行")
    
    print(f"\n✅ 合并完成！文件已保存到: {output_path}")

if __name__ == "__main__":
    merge_csv_files()

