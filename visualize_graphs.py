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

    
    返回
    ----
    A: 邻接矩阵
    var_names: 变量名列表
    """
    A = np.load(adj_matrix_path)
    with open(variables_path, 'r', encoding='utf-8') as f:
        var_names = [line.strip() for line in f if line.strip()]
    return A, var_names


def resolve_node_overlaps(
    pos: dict[int, np.ndarray],
    node_size: int,
    fixed_nodes: set[int] | None = None,
    max_iter: int = 600,
) -> dict[int, np.ndarray]:
    """
    迭代分离重叠节点，保证节点圆圈不互相覆盖。
    """
    if not pos:
        return pos

    fixed_nodes = fixed_nodes or set()
    nodes = list(pos.keys())
    node_radius = max(0.2, np.sqrt(node_size) / 40.0)
    min_dist = node_radius * 2.15  # 留一点额外间隙，避免视觉上贴边
    eps = 1e-9

    for _ in range(max_iter):
        moved = False
        for i in range(len(nodes)):
            n1 = nodes[i]
            p1 = pos[n1]
            for j in range(i + 1, len(nodes)):
                n2 = nodes[j]
                p2 = pos[n2]
                delta = p2 - p1
                dist = float(np.linalg.norm(delta))
                if dist >= min_dist:
                    continue

                moved = True
                if dist < eps:
                    # 完全重合时给一个固定方向扰动，保证可复现
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta / dist

                overlap = min_dist - max(dist, eps)
                shift = direction * (overlap / 2.0 + 1e-4)

                n1_fixed = n1 in fixed_nodes
                n2_fixed = n2 in fixed_nodes
                if n1_fixed and n2_fixed:
                    continue
                if n1_fixed:
                    pos[n2] = p2 + shift * 2.0
                elif n2_fixed:
                    pos[n1] = p1 - shift * 2.0
                else:
                    pos[n1] = p1 - shift
                    pos[n2] = p2 + shift

        if not moved:
            break

    return pos


def fit_label_into_circle(raw_label: str, node_size: int, target_font_size: int) -> tuple[str, int]:
    """
    自动换行并调整字号，尽量保证标签完整显示在节点圆圈内部。
    """
    if not raw_label:
        return "", target_font_size

    normalized = raw_label.replace(":", "_")
    chunks = [chunk for chunk in normalized.split("_") if chunk]
    if not chunks:
        chunks = [normalized]

    diameter_pts = 2.0 * np.sqrt(node_size / np.pi)
    for font in range(target_font_size, 2, -1):
        char_width_pts = max(1.0, 0.58 * font)
        line_height_pts = max(1.0, 1.2 * font)
        max_chars = max(3, int(diameter_pts * 0.82 / char_width_pts))
        max_lines = max(1, int(diameter_pts * 0.82 / line_height_pts))

        lines: list[str] = []
        current = ""
        for chunk in chunks:
            candidate = chunk if not current else f"{current}_{chunk}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                while len(chunk) > max_chars:
                    lines.append(chunk[:max_chars])
                    chunk = chunk[max_chars:]
                current = chunk
        if current:
            lines.append(current)

        if len(lines) <= max_lines:
            return "\n".join(lines), font

    # 极端情况下仍无法完整放入，退化为最小字号并截断
    min_font = 3
    char_width_pts = max(1.0, 0.58 * min_font)
    line_height_pts = max(1.0, 1.2 * min_font)
    max_chars = max(3, int(diameter_pts * 0.82 / char_width_pts))
    max_lines = max(1, int(diameter_pts * 0.82 / line_height_pts))
    text = normalized
    wrapped = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    wrapped = wrapped[:max_lines]
    if wrapped and len("".join(wrapped)) < len(text):
        wrapped[-1] = wrapped[-1][:-1] + "…"
    return "\n".join(wrapped), min_font


def plot_dag(
    A: np.ndarray,
    var_names: list[str],
    output_path: Path,
    title: str = "因果图",
    figsize: tuple[int, int] = (20, 16),
    node_size: int = 3500,
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
    layout: 布局算法 ('spring', 'hierarchical', 'circular', 'kamada_kawai', 'score_centered')
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 找出所有边（应用阈值过滤）
    edges = []
    edge_weights = []
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if i == j:
                continue  # 跳过自环边，避免自己连自己
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
            # 约定 M[target, source]：A[i,j] 中 source=j -> target=i
            G.add_edge(new_j, new_i, weight=abs(A[i, j]))
        
        # 使用过滤后的变量名
        display_var_names = filtered_var_names
    else:
        # 显示所有节点
        for i, var_name in enumerate(var_names):
            G.add_node(i, label=var_name)
        
        # 添加边
        for i, j in edges:
            # 约定 M[target, source]：A[i,j] 中 source=j -> target=i
            G.add_edge(j, i, weight=abs(A[i, j]))
        
        # 使用原始变量名
        display_var_names = var_names
        node_mapping = {i: i for i in range(len(var_names))}  # 单位映射
    
    if len(edges) == 0:
        print(f"⚠️  警告: {title} 中没有边，跳过绘图")
        return
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 选择布局
    fixed_nodes: set[int] = set()
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
    elif layout == "score_centered":
        # 以 score 为中心，其他节点按与 score 的最短路径距离分层环绕
        # 同时加入可复现的随机扰动，避免过于规则整齐
        score_nodes = [node for node, data in G.nodes(data=True) if data.get("label") == "score"]
        if score_nodes:
            score_node = score_nodes[0]
            fixed_nodes.add(score_node)
            pos = {score_node: np.array([0.0, 0.0])}
            other_nodes = [node for node in G.nodes() if node != score_node]

            if other_nodes:
                UG = G.to_undirected()
                distances = nx.single_source_shortest_path_length(UG, score_node)
                default_layer = max(distances.values(), default=0) + 1

                layers: dict[int, list[int]] = {}
                for node in other_nodes:
                    layer = distances.get(node, default_layer)
                    layers.setdefault(layer, []).append(node)

                base_radius = 1.8
                radius_gap = 1.25
                for layer in sorted(layers):
                    nodes_in_layer = sorted(layers[layer], key=lambda n: G.degree(n), reverse=True)
                    count = len(nodes_in_layer)
                    radius = base_radius + (layer - 1) * radius_gap
                    rng = np.random.default_rng(42 + layer)
                    global_angle_offset = rng.uniform(0, 2 * np.pi)
                    for idx, node in enumerate(nodes_in_layer):
                        base_angle = 2 * np.pi * idx / max(count, 1)
                        angle_jitter = rng.normal(0.0, 0.22)
                        radial_jitter = rng.normal(0.0, 0.35)
                        x_jitter = rng.normal(0.0, 0.12)
                        y_jitter = rng.normal(0.0, 0.12)
                        angle = base_angle + global_angle_offset + angle_jitter
                        node_radius = max(0.8, radius + radial_jitter)
                        pos[node] = np.array(
                            [
                                node_radius * np.cos(angle) + x_jitter,
                                node_radius * np.sin(angle) + y_jitter,
                            ]
                        )
        else:
            pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 防止节点圆圈彼此重叠；score_centered 下 score 节点会保持在中心
    pos = resolve_node_overlaps(pos, node_size=node_size, fixed_nodes=fixed_nodes)
    
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
    
    # 绘制标签（自动换行并保证尽量落在圈内）
    labels = {}
    label_font_sizes = {}
    score_node_ids = set()
    for new_idx, var_name in enumerate(display_var_names):
        if var_name == "score":
            base_label = "score"
            score_node_ids.add(new_idx)
        elif var_name.startswith("r_"):
            base_label = var_name[2:]  # 去掉 "r_" 前缀
        elif var_name.startswith("active_"):
            base_label = "A:" + var_name[7:]  # 去掉 "active_" 前缀并保留类型前缀
        else:
            base_label = var_name
        wrapped_label, actual_font_size = fit_label_into_circle(
            base_label,
            node_size=node_size,
            target_font_size=font_size,
        )
        labels[new_idx] = wrapped_label
        label_font_sizes[new_idx] = actual_font_size

    for node_id, text in labels.items():
        x, y = pos[node_id]
        text_font_size = label_font_sizes[node_id]
        if node_id in score_node_ids:
            text_font_size *= 1.35  # 单独放大 score 字体
        plt.text(
            x,
            y,
            text,
            fontsize=text_font_size,
            fontweight='bold',
            fontfamily='sans-serif',
            ha='center',
            va='center',
            zorder=5,
        )
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存图片: {output_path}")


def visualize_all_graphs(output_dir: Path, edge_threshold: float = 1e-6) -> None:
    """
    仅可视化全局因果图（不生成各环境的 png/npy）。

    参数
    ----
    output_dir: 输出目录路径
    edge_threshold: 边阈值，只显示权重大于等于此阈值的边（用于全局图）
    """
    print("=" * 60)
    print("开始可视化全局因果图")
    print("=" * 60)

    # 仅可视化全局图
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
            font_size=12,  # 放大圈内字体
            layout="score_centered",  # 以 score 为中心，其他节点环绕分层
            filter_isolated=True,  # 总图过滤孤立节点
            edge_threshold=edge_threshold,  # 应用阈值过滤
        )
    else:
        print(f"⚠️  未找到全局图文件: {global_A_path} 或 {global_vars_path}")

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"图片已保存: {output_dir / 'graph_global.png'}")


if __name__ == "__main__":
    from pathlib import Path
    output_dir = Path(__file__).resolve().parent / "output"
    visualize_all_graphs(output_dir)

