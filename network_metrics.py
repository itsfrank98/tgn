import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import yaml
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import random
with open("parameters.yaml", 'r') as params_file:
    args = yaml.safe_load(params_file)


net_params = args["network_metrics_params"]
real_network_src = net_params["real_network_src"]
synthetic_network_src = net_params["synthetic_network_src"]
threshold = net_params["threshold"]
compute_metrics = net_params["compute_metrics"]
draw_network = net_params["draw_network"]
nodes_to_sample = net_params["sample_n_nodes"]
group_layout = net_params["group_layout"]
#synthetic = net_params["synthetic"]

def compute_network_metrics(G: nx.DiGraph):
    print("=" * 40)
    print("       NETWORK METRICS SUMMARY")
    print("=" * 40)
    # --- Basic Info ---
    print("\n📌 Basic Info")
    print(f"  Nodes:          {G.number_of_nodes()}")
    print(f"  Edges:          {G.number_of_edges()}")
    # --- Degree ---
    print("\n📊 Degree Statistics")
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    degrees = [d for _, d in G.degree()]
    print(f"  Avg Degree:         {np.mean(degrees):.4f}")
    print(f"  Avg In-Degree:      {np.mean(in_degrees):.4f}")
    print(f"  Avg Out-Degree:     {np.mean(out_degrees):.4f}")
    print(f"  Max In-Degree:      {max(in_degrees)}")
    print(f"  Max Out-Degree:     {max(out_degrees)}")
    # --- Density ---
    print("\n🔗 Connectivity")
    print(f"  Density:            {nx.density(G):.4f}")
    print(f"  Strongly Connected: {nx.is_strongly_connected(G)}")
    print(f"  Weakly Connected:   {nx.is_weakly_connected(G)}")
    print(f"  Num SCC:            {nx.number_strongly_connected_components(G)}")
    print(f"  Num WCC:            {nx.number_weakly_connected_components(G)}")
    # --- Centrality ---
    print("\n🎯 Centrality (averages)")
    in_centrality = nx.in_degree_centrality(G)
    print(f"  Avg In-Degree Centrality:   {np.mean(list(in_centrality.values())):.4f}")
    out_centrality = nx.out_degree_centrality(G)
    print(f"  Avg Out-Degree Centrality:  {np.mean(list(out_centrality.values())):.4f}")
    # --- Path lengths (on largest SCC for validity) ---
    print("\n📏 Path Lengths (largest SCC)")
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    SCC = G.subgraph(largest_scc)
    avg_path = nx.average_shortest_path_length(SCC)
    print(f"  Avg Shortest Path Length:  {avg_path:.4f}")
    communities = louvain_communities(G, seed=123)
    modularity = nx.community.modularity(G, communities)
    print(f"  Modularity: {modularity}")

    print("\n" + "=" * 40)


def grouped_layout(G, group_lookup, spread=0.15, seed=42):
    """
    Places nodes in circular zones by group.
    - Each group gets a 'center' point on a large circle
    - Nodes within a group are scattered around that center
    """
    rng = np.random.default_rng(seed)

    groups = list(dict.fromkeys(group_lookup.values()))  # preserve order, deduplicate
    n_groups = len(groups)

    # Assign each group a center point on a circle
    group_centers = {
        group: np.array([
            np.cos(2 * np.pi * i / n_groups),
            np.sin(2 * np.pi * i / n_groups)
        ])
        for i, group in enumerate(groups)
    }

    pos = {}
    for node in G.nodes():
        group = group_lookup.get(int(node), "unknown")
        center = group_centers[group]
        # Scatter node randomly within the group's zone
        offset = rng.uniform(-spread, spread, size=2)
        pos[node] = center + offset

    return pos, group_centers

def plot_network(G, color_map, leanings_lookup, title, group_layout=False):
    node_colors = [color_map[leanings_lookup[int(node)]] for node in G.nodes()]
    if group_layout:
        pos, group_centers = grouped_layout(G, leanings_lookup, spread=0.15)

        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx(
            G, pos=pos, node_color=node_colors, node_size=200, font_color="white", font_weight="bold",
            edge_color="#cccccc", width=1.5, with_labels=False, ax=ax
        )
        for group, center in group_centers.items():
            ax.text(
                center[0], center[1] + 0.22,  # offset above the cluster
                group,
                fontsize=11, fontweight="bold", ha="center", va="center",
                color=color_map.get(group, "black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    else:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, k=1, seed=42)
        pos = nx.circular_layout(G)


        legend_elements = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
        plt.legend(handles=legend_elements, loc="best", fontsize=11)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


eel = []
real_network = []
nodes_with_connections = set()
G = nx.DiGraph()
if synthetic_network_src:
    with open(synthetic_network_src, "rb") as f:
        el = pickle.load(f)
    eel += [(int(e[0]), int(e[1])) for e in el if e[2] > threshold]


if real_network_src:
    with open(real_network_src, "rb") as f:
        for l in f.readlines():
            e1, e2 = l.split()
            e1 = int(e1.strip())
            e2 = int(e2.strip())
            eel.append((e1, e2))
            nodes_with_connections.add(e1)
            nodes_with_connections.add(e2)

G.add_edges_from(eel)

if compute_metrics:
    compute_network_metrics(G)
if draw_network:
    color_mapping = {
        "far-left": "black", "far-right": "red", "right": "orange",
        "left": "blue", "center": "pink", "non-political": "green", "unknown": "magenta"
    }

    if synthetic_network_src:
        title = "Synthetic network by political leaning"
        df = pd.read_csv("data/clean_data/synthetic_posts.csv", index_col=0)
    else:
        title = "Real network by political leaning"
        df = pd.read_csv("data/clean_data/stance_mistral_real.csv", index_col=0)

    leanings = df.drop_duplicates(subset="political_leaning")["political_leaning"].tolist()
    df_nodes = df["account_id"].drop_duplicates().tolist()
    G_new = G.copy()
    for node in G.nodes():
        if node not in df_nodes or (real_network_src and node not in nodes_with_connections):
            G_new.remove_node(node)
    if nodes_to_sample:
        sampled_nodes = random.sample(list(G_new.nodes()), nodes_to_sample)
        G_new = G_new.subgraph(sampled_nodes)
    G = G_new.copy()
    isolated_nodes = list(nx.isolates(G_new))
    G.remove_nodes_from(isolated_nodes)

    print(len(G))

    leanings_lookup = df.set_index("account_id")["political_leaning"].to_dict()
    plot_network(G, group_layout=group_layout, color_map=color_mapping, leanings_lookup=leanings_lookup, title=title+f", sampled {G.number_of_nodes()} nodes")
