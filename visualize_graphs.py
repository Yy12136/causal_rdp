"""
可视化因果图模块
使用 networkx 和 matplotlib 绘制有向图
"""

from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def load_graph(adj_matrix_path: Path, variables_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    加载邻接矩阵和变量列表
    
    参数
    ----
    adj_matrix_path: 邻接矩阵 .npy 文件路径
    variables_path: 变量列表 .txt 文件路径
    
    返回
    ----
    A: 邻接矩阵
    var_names: 变量名列表
    """
    A = np.load(adj_matrix_path)
    with open(variables_path, 'r', encoding='utf-8') as f:
        var_names = [line.strip() for line in f if line.strip()]
    return A, var_names


def plot_dag(
    A: np.ndarray,
    var_names: list[str],
    output_path: Path,
    title: str = "因果图",
    figsize: tuple[int, int] = (20, 16),
    node_size: int = 800,
    font_size: int = 8,
    edge_width_scale: float = 1.0,
    layout: str = "spring",
    filter_isolated: bool = False,
    edge_threshold: float = 1e-6,
) -> None:
    """
    绘制有向无环图（DAG）
    
    参数
    ----
    A: 邻接矩阵 (n x n)
    var_names: 变量名列表
    output_path: 输出图片路径
    title: 图标题
    figsize: 图片大小
    node_size: 节点大小
    font_size: 字体大小
    edge_width_scale: 边宽度缩放因子
    layout: 布局算法 ('spring', 'hierarchical', 'circular', 'kamada_kawai')
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 找出所有边（应用阈值过滤）
    edges = []
    edge_weights = []
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if abs(A[i, j]) >= edge_threshold:  # 只保留大于等于阈值的边
                edges.append((i, j))
                edge_weights.append(abs(A[i, j]))
    
    # 根据 filter_isolated 参数决定是否过滤孤立节点
    if filter_isolated:
        # 只显示有边的节点
        nodes_with_edges = set()
        for i, j in edges:
            nodes_with_edges.add(i)
            nodes_with_edges.add(j)
        
        # 创建节点映射
        node_mapping = {}  # 原索引 -> 新索引的映射
        filtered_var_names = []
        for idx, var_name in enumerate(var_names):
            if idx in nodes_with_edges:
                new_idx = len(filtered_var_names)
                node_mapping[idx] = new_idx
                filtered_var_names.append(var_name)
                G.add_node(new_idx, label=var_name, original_idx=idx)
        
        # 添加边（使用新的节点索引）
        for i, j in edges:
            new_i = node_mapping[i]
            new_j = node_mapping[j]
            G.add_edge(new_i, new_j, weight=abs(A[i, j]))
        
        # 使用过滤后的变量名
        display_var_names = filtered_var_names
    else:
        # 显示所有节点
        for i, var_name in enumerate(var_names):
            G.add_node(i, label=var_name)
        
        # 添加边
        for i, j in edges:
            G.add_edge(i, j, weight=abs(A[i, j]))
        
        # 使用原始变量名
        display_var_names = var_names
        node_mapping = {i: i for i in range(len(var_names))}  # 单位映射
    
    if len(edges) == 0:
        print(f"⚠️  警告: {title} 中没有边，跳过绘图")
        return
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 选择布局
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "hierarchical":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 根据变量类型设置节点颜色
    node_colors = []
    for var_name in display_var_names:
        if var_name == "score":
            node_colors.append("#FF6B6B")  # 红色 - score
        elif var_name.startswith("r_"):
            node_colors.append("#4ECDC4")  # 青色 - reward
        elif var_name.startswith("active_"):
            node_colors.append("#95E1D3")  # 浅青色 - active
        else:
            node_colors.append("#FFE66D")  # 黄色 - 其他
    
    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5
    )
    
    # 绘制边（根据权重设置宽度）
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        if max_weight > min_weight:
            edge_widths = [
                (w - min_weight) / (max_weight - min_weight) * 2 + 0.5
                for w in edge_weights
            ]
        else:
            edge_widths = [1.0] * len(edge_weights)
        edge_widths = [w * edge_width_scale for w in edge_widths]
    else:
        edge_widths = [1.0] * len(edges)
    
    # 绘制边（G中的边已经使用新索引）
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1'
    )
    
    # 绘制标签（简化变量名以节省空间）
    labels = {}
    for new_idx, var_name in enumerate(display_var_names):
        # 简化标签：只保留关键部分
        if var_name == "score":
            labels[new_idx] = "score"
        elif var_name.startswith("r_"):
            labels[new_idx] = var_name[2:][:15]  # 去掉 "r_" 前缀，最多15字符
        elif var_name.startswith("active_"):
            labels[new_idx] = "A:" + var_name[7:][:12]  # 去掉 "active_" 前缀，最多12字符
        else:
            labels[new_idx] = var_name[:15]
    
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=font_size,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # 设置标题
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='score'),
        Patch(facecolor='#4ECDC4', label='r_* (reward)'),
        Patch(facecolor='#95E1D3', label='active_*'),
        Patch(facecolor='#FFE66D', label='other'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 添加统计信息
    num_nodes_display = len(display_var_names)
    num_nodes_total = len(var_names)
    num_edges = len(edges)
    if edge_threshold > 1e-6:
        info_text = f"Nodes: {num_nodes_display} | Edges: {num_edges} (threshold ≥ {edge_threshold})"
    elif filter_isolated and num_nodes_display < num_nodes_total:
        info_text = f"Nodes: {num_nodes_display}/{num_nodes_total} (with edges) | Edges: {num_edges}"
    else:
        info_text = f"Nodes: {num_nodes_display} | Edges: {num_edges}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存图片: {output_path}")


def visualize_all_graphs(output_dir: Path, edge_threshold: float = 1e-6) -> None:
    """
    可视化 output_dir 中的所有图
    
    参数
    ----
    output_dir: 输出目录路径
    edge_threshold: 边阈值，只显示权重大于等于此阈值的边（用于全局图）
    """
    print("=" * 60)
    print("开始可视化所有因果图")
    print("=" * 60)
    print("\n说明：")
    print("  - 全局图：使用所有环境数据训练得到的完整因果图")
    print("  - 子图：基于全局图，针对每个环境裁剪得到的子图")
    print("    （移除在该环境下为常量的变量及其边）")
    
    # 1. 可视化全局图
    global_A_path = output_dir / "A_dagma_global.npy"
    global_vars_path = output_dir / "variables_global.txt"
    
    if global_A_path.exists() and global_vars_path.exists():
        print(f"\n正在可视化全局图...")
        print(f"  应用边阈值: {edge_threshold}")
        A_global, var_names = load_graph(global_A_path, global_vars_path)
        plot_dag(
            A_global, var_names,
            output_dir / "graph_global.png",
            title=f"Global Causal Graph (Edge Threshold ≥ {edge_threshold})",
            figsize=(24, 18),
            font_size=7,
            filter_isolated=True,  # 总图过滤孤立节点
            edge_threshold=edge_threshold,  # 应用阈值过滤
        )
    else:
        print(f"⚠️  未找到全局图文件: {global_A_path} 或 {global_vars_path}")
    
    # 2. 可视化所有环境子图
    env_A_files = sorted(output_dir.glob("A_dagma_env_*.npy"))
    
    if len(env_A_files) == 0:
        print("⚠️  未找到环境子图文件")
    else:
        print(f"\n找到 {len(env_A_files)} 个环境子图")
        
        for env_A_path in env_A_files:
            # 提取环境名称
            env_name = env_A_path.stem.replace("A_dagma_env_", "")
            env_vars_path = output_dir / f"variables_env_{env_name}.txt"
            
            if not env_vars_path.exists():
                print(f"⚠️  未找到变量文件: {env_vars_path}")
                continue
            
            print(f"  正在可视化环境: {env_name}...")
            A_env, var_names = load_graph(env_A_path, env_vars_path)
            # 将环境名称转换为更友好的格式
            env_name_display = env_name.replace("_", " ").replace("shadow hand", "Shadow Hand")
            plot_dag(
                A_env, var_names,
                output_dir / f"graph_env_{env_name}.png",
                title=f"Environment Subgraph: {env_name_display} (Pruned from Global Graph)",
                figsize=(20, 16),
                font_size=7,
                filter_isolated=False,  # 子图不过滤，显示所有节点
            )
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"所有图片已保存到: {output_dir}")
    print(f"  - graph_global.png: 全局因果图")
    print(f"  - graph_env_*.png: 各环境子图")


if __name__ == "__main__":
    from pathlib import Path
    output_dir = Path(__file__).resolve().parent / "output"
    visualize_all_graphs(output_dir)

