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
synthetic = net_params["synthetic"]

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

def plot_network(G, color_map, leanings_lookup, title):
    node_colors = [color_map[leanings_lookup[int(node)]] for node in G.nodes()]

    # --- Plot ---
    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G, k=1, seed=42)

    nx.draw_networkx(G, pos=pos, node_color=node_colors, node_size=200, font_color="white", font_weight="bold",
        edge_color="#cccccc", width=1.5, with_labels=False
    )

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
    color_mapping = {"far-left": "black", "far-right": "red", "right": "orange", "left": "blue", "center": "pink", "non-political": "green", "unknown": "magenta"}
    if synthetic:
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
    plot_network(G, color_map=color_mapping, leanings_lookup=leanings_lookup, title="Synthetic Network by political leaning")
