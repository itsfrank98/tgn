import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import yaml
import pickle

with open("parameters.yaml", 'r') as params_file:
    args = yaml.safe_load(params_file)

net_params = args["network_metrics_params"]
real_net_src = net_params["real_network_src"]
threshold = net_params["threshold"]
synthetic_network_src = f"connections_{args["negative_sampling_ratio"]}.pkl"

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
    # --- Clustering ---
    """print("\n🔺 Clustering")
    avg_clustering = nx.average_clustering(G)  # uses directed clustering by default
    print(f"  Avg Clustering Coeff (directed): {avg_clustering:.4f}")"""
    # --- Centrality ---
    print("\n🎯 Centrality (averages)")
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        print(f"  Avg Eigenvector Centrality: {np.mean(list(eigenvector.values())):.4f}")
    except nx.PowerIterationFailedConvergence:
        print("  Avg Eigenvector Centrality: (did not converge)")
    print(f"  Avg In-Degree Centrality:   {np.mean(list(in_centrality.values())):.4f}")
    print(f"  Avg Out-Degree Centrality:  {np.mean(list(out_centrality.values())):.4f}")
    print(f"  Avg Betweenness Centrality: {np.mean(list(betweenness.values())):.4f}")
    print(f"  Avg Closeness Centrality:   {np.mean(list(closeness.values())):.4f}")
    # --- Path lengths (on largest SCC for validity) ---
    print("\n📏 Path Lengths (largest SCC)")
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    SCC = G.subgraph(largest_scc)
    #if len(SCC) > 1:
    avg_path = nx.average_shortest_path_length(SCC)
    diameter = nx.diameter(SCC)
    print(f"  Avg Shortest Path Length:  {avg_path:.4f}")
    print(f"  Diameter:                  {diameter}")
    communities = louvain_communities(G, seed=123)
    modularity = nx.community.modularity(G, communities)
    print(f"  Modularity: {modularity}")

    print("\n" + "=" * 40)

with open(synthetic_network_src, "rb") as f:
    el = pickle.load(f)


real_network = []
if real_net_src != "None":
    with open(real_net_src, "rb") as f:
        for l in f.readlines():
            e1, e2 = l.split()
            e1 = int(e1.strip())
            e2 = int(e2.strip())
            real_network.append((e1, e2))

eel = [(e[0], e[1]) for e in el if e[2] > threshold]
complete_network = real_network + eel
G = nx.DiGraph()
G.add_edges_from(eel)
compute_network_metrics(G)