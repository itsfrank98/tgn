import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import yaml
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from collections import defaultdict
import random

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
    communities = louvain_communities(G, seed=123)
    modularity = nx.community.modularity(G, communities)
    print(f"  Modularity: {modularity}")

    print("\n" + "=" * 40)

def stratified_sample(df: pd.DataFrame, n, label_col="political_leaning", min_per_label=2, random_state=42):
    labels = df[label_col].unique()
    n_labels = len(labels)
    if n < n_labels * min_per_label:
        raise ValueError(f"n={n} is too small to guarantee {min_per_label} samples per label ({n_labels} labels)")

    # Floor allocation
    floor_alloc = {label: min_per_label for label in labels}
    remaining = n - n_labels * min_per_label

    # Proportional allocation for the remainder
    counts = df[label_col].value_counts()
    proportions = counts / counts.sum()
    prop_alloc = (proportions * remaining).round().astype(int)

    # Fix rounding drift
    diff = remaining - prop_alloc.sum()
    if diff != 0:
      # Add/remove from the largest groups
      adjustment_labels = prop_alloc.nlargest(abs(diff)).index
      prop_alloc[adjustment_labels] += int(np.sign(diff))

    # Combine and sample
    samples = []
    for label in labels:
      total = floor_alloc[label] + prop_alloc.get(label, 0)
      group = df[df[label_col] == label]
      total = min(total, len(group))  # can't sample more than available
      samples.append(group.sample(n=total, random_state=random_state))

    return pd.concat(samples).sample(frac=1, random_state=random_state).reset_index(drop=True)

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
        group = group_lookup.get(int(node)) #, "unknown"
        center = group_centers[group]
        # Scatter node randomly within the group's zone
        offset = rng.uniform(-spread, spread, size=2)
        pos[node] = center + offset

    return pos, group_centers

def grouped_circular_layout(G, group_lookup, group_arc_gap=0.08, seed=42, order=None):
    """
    Places nodes along a big circle, grouped by label.
    Each group occupies a contiguous arc, with small gaps between groups.
    Nodes within a group are spread along their arc with slight radial jitter.
    """
    rng = np.random.default_rng(seed)

    # Count nodes per group (preserving order)
    if order :
        groups = order
    else:
        groups = list(dict.fromkeys(group_lookup.values()))
    group_counts = {g: sum(1 for v in group_lookup.values() if v == g) for g in groups}
    total_nodes = sum(group_counts.values())
    n_groups = len(groups)

    # Total angle used by gaps
    total_gap = group_arc_gap * n_groups  # radians
    total_arc = 2 * np.pi - total_gap     # remaining angle for nodes

    # Assign arc length to each group proportional to node count
    group_arc_lengths = {g: (group_counts[g] / total_nodes) * total_arc for g in groups}

    # Compute start angle for each group
    group_start_angles = {}
    current_angle = 0.0
    for g in groups:
        group_start_angles[g] = current_angle
        current_angle += group_arc_lengths[g] + group_arc_gap

    # Assign angles to individual nodes within each group
    group_node_lists = {g: [] for g in groups}
    for node in G.nodes():
        g = group_lookup.get(int(node)) #, "unknown"
        if g != "unknown":
            group_node_lists[g].append(node)

    radius = 1.0
    radial_jitter = 0.04  # slight inward/outward jitter

    pos = {}
    group_centers = {}

    for g in groups:
        nodes = group_node_lists[g]
        n = len(nodes)
        start = group_start_angles[g]
        arc = group_arc_lengths[g]

        # Center angle for label placement
        mid_angle = start + arc / 2
        group_centers[g] = np.array([np.cos(mid_angle), np.sin(mid_angle)])

        for i, node in enumerate(nodes):
            # Spread nodes evenly along the arc
            if n > 1:
                angle = start + (i / (n - 1)) * arc
            else:
                angle = start + arc / 2

            r = radius + rng.uniform(-radial_jitter, radial_jitter)
            pos[node] = np.array([r * np.cos(angle), r * np.sin(angle)])

    return pos, group_centers

def plot_network(G, color_map, leanings_lookup, title, group_layout=False, order=None):
    node_colors = [color_map[leanings_lookup[int(node)]] for node in G.nodes()]
    if group_layout=="grouped":
        pos, group_centers = grouped_layout(G, leanings_lookup, spread=0.15)
        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx(
            G, pos=pos, node_color=node_colors, node_size=200, font_color="white", font_weight="bold",
            edge_color="#cccccc", width=0.1, with_labels=False, ax=ax
        )
        for group, center in group_centers.items():
            ax.text(
                center[0], center[1] + 0.22,  # offset above the cluster
                group,
                fontsize=11, fontweight="bold", ha="center", va="center",
                color=color_map.get(group, "black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    elif group_layout=="circular":
        pos, group_centers = grouped_circular_layout(G, leanings_lookup, order=order)

        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx(
            G, pos=pos, node_color=node_colors, node_size=200, edge_color="#cccccc", width=1, alpha=0.6,
            with_labels=False, ax=ax
        )

        # Place labels just outside the circle at each group's midpoint
        label_radius = 1.22
        for group, center in group_centers.items():
            # Normalize center to place label outside the ring
            angle = np.arctan2(center[1], center[0])
            lx = label_radius * np.cos(angle)
            ly = label_radius * np.sin(angle)
            ax.text(
                lx, ly, group,
                fontsize=11, fontweight="bold", ha="center", va="center",
                color=color_map.get(group, "black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    else:
        plt.figure(figsize=(8, 6))
        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos=pos, node_color=node_colors, node_size=200, font_color="white", font_weight="bold",
                         edge_color="#cccccc", width=8, with_labels=False)
        legend_elements = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
        plt.legend(handles=legend_elements, loc="best", fontsize=11)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def fit_real_stats(real_profiles: pd.DataFrame, real_graph: nx.DiGraph, political_leaning_key: str) -> dict:
    """
    For each (source_group, target_group) pair, computes:
      - p_connect        : P(a user in source_group has >= 1 followee in target_group)
      - nonzero_degrees  : empirical distribution of # followees in target_group,
                           conditional on having at least one (used to sample N)
    """
    political_leanings = real_profiles[political_leaning_key].drop_duplicates().tolist()
    id_to_group = dict(zip(real_profiles["account_id"].astype(str), real_profiles[political_leaning_key]))

    stats = {}
    for src_group in political_leanings:
        members = {str(m) for m in real_profiles.loc[real_profiles[political_leaning_key] == src_group, "account_id"]}

        # cross_counts[tgt_group] = list of per-member followee counts toward tgt_group
        cross_counts = defaultdict(list)
        for user in members:
            per_target = defaultdict(int)
            if user in real_graph:
                for f in real_graph.successors(user):
                    tg = id_to_group.get(f)
                    if tg:
                        per_target[tg] += 1
            for tg in political_leanings:
                cross_counts[tg].append(per_target.get(tg, 0))

        src_stats = {}
        for tgt_group in political_leanings:
            counts = cross_counts[tgt_group]
            p_connect = float(np.mean([1 if c > 0 else 0 for c in counts]))
            nonzero = [c for c in counts if c > 0]
            nonzero_degrees = np.array(nonzero, dtype=int) if nonzero else np.array([1])
            src_stats[tgt_group] = {
                "p_connect": p_connect,
                "nonzero_degrees": nonzero_degrees,
            }

        stats[src_group] = src_stats
        print(f"  [{src_group}]  " + "  ".join(
            f"{tg}:{src_stats[tg]['p_connect']:.2f}" for tg in political_leanings
        ))

    return stats


def build_synthetic_edges(synth_profiles: pd.DataFrame, political_leaning_key: str, connections: list, real_stats: dict, seed=42) -> list:
    """
    For each synthetic user u1 with leaning P, and for each target group L:
      1. Draw Bernoulli(p_connect[P][L]) — skip group L if 0.
      2. Sample N from nonzero_degrees[P][L].
      3. Take the top-N scored candidates from group L.

    connections : list of (u1_id, u2_id, score)
    """
    rng = np.random.default_rng(seed)

    id_to_group = dict(zip(synth_profiles["account_id"].astype(str), synth_profiles[political_leaning_key]))

    # per-user candidate lists per target group, sorted by score descending
    user_group_candidates: dict = defaultdict(lambda: defaultdict(list))
    for u1, u2, score in connections:
        u1, u2 = str(u1), str(u2)
        tg = id_to_group.get(u2)
        if tg:
            user_group_candidates[u1][tg].append((score, u2))
    for u1 in user_group_candidates:
        for tg in user_group_candidates[u1]:
            user_group_candidates[u1][tg].sort(reverse=True)

    edges = []
    for u1, src_group in id_to_group.items():
        if src_group not in real_stats:
            continue
        for tgt_group, pair_stats in real_stats[src_group].items():
            # Bernoulli: does this user connect to tgt_group at all?
            if rng.random() > pair_stats["p_connect"]:
                continue
            n = int(rng.choice(pair_stats["nonzero_degrees"])) * 4
            pool = user_group_candidates[u1][tgt_group]
            edges.extend((u1, u2) for _, u2 in pool[:n])

    return [(u, v) for u, v in set(edges) if u != v]


if __name__ == "__main__":
    with open("parameters.yaml", 'r') as params_file:
        args = yaml.safe_load(params_file)

    net_params = args["net_analysis"]
    draw_network = net_params["draw_network"]
    compute_metrics = net_params["compute_metrics"]
    mirror_degree_distribution = net_params["mirror_degree_distribution"]

    real_edges_src = net_params["real_edges_src"]
    synthetic_edges_src = net_params["synthetic_edges_src"]
    threshold = net_params["threshold"]

    real_network = []
    synthetic_network = []
    political_leaning_key = "political_leaning"

    nodes_with_connections = set()
    Real_graph = nx.DiGraph()
    Synthetic_graph = nx.DiGraph()

    if real_edges_src:
        with open(real_edges_src, "r") as f:
            for l in f.readlines():
                e1, e2 = l.split("\t")
                e1, e2 = e1.strip(), e2.strip()
                real_network.append((e1, e2))
                nodes_with_connections.add(e1)
                nodes_with_connections.add(e2)

        Real_graph.add_edges_from(real_network)

    if synthetic_edges_src and not mirror_degree_distribution:
        with open(synthetic_edges_src, "rb") as f:
            el = pickle.load(f)
            if synthetic_edges_src.__contains__("filtered"):
                synthetic_network = [(str(e[0]), str(e[1])) for e in el]
            else:
                synthetic_network = [(str(e[0]), str(e[1])) for e in el if e[2]>threshold]

        Synthetic_graph.add_edges_from(synthetic_network)

    if compute_metrics:
        if len(Real_graph) > 0:
            print("REAL NETWORK STATS")
            compute_network_metrics(Real_graph)
        if len(Synthetic_graph) > 0:
            print("\n SYNTHETIC NETWORK STATS")
            compute_network_metrics(Synthetic_graph)

    if mirror_degree_distribution:
        output_edges_path = net_params["mirroring_node_degrees_params"]["output_edges_path"]
        real_profiles_src = net_params["mirroring_node_degrees_params"]["real_profiles_src"]
        synthetic_profiles_src = net_params["mirroring_node_degrees_params"]["synthetic_profiles_src"]

        real_profiles = pd.read_csv(real_profiles_src)
        synthetic_profiles = pd.read_csv(synthetic_profiles_src).drop_duplicates(subset="account_id")
        with open(synthetic_edges_src, "rb") as f:
            synthetic_network = pickle.load(f)
        synthetic_network = [e for e in synthetic_network if e[2]>threshold]
        print(len(synthetic_network))

        # ── fit real stats ────────────────────────────────────────────────────────
        print("\nFitting real-network statistics …")
        real_stats = fit_real_stats(real_profiles, Real_graph, political_leaning_key=political_leaning_key)

        # ── edge selection ────────────────────────────────────────────────────────
        print("\nSelecting edges …")
        synthetic_edges = build_synthetic_edges(synthetic_profiles, political_leaning_key=political_leaning_key, connections=synthetic_network, real_stats=real_stats)
        print(f"  Synthetic edges created: {len(synthetic_edges)}")

        # ── build synthetic graph ─────────────────────────────────────────────────
        synth_graph = nx.DiGraph()
        #synth_graph.add_nodes_from(synthetic_profiles["account_id"])
        synth_graph.add_edges_from(synthetic_edges)

        # ── save edge list ────────────────────────────────────────────────────────
        with open(output_edges_path, "wb") as f:
            pickle.dump(synthetic_edges, f)

        print(f"\nSynthetic edge list saved to: {output_edges_path}")

        # ── fit real stats ────────────────────────────────────────────────────────
        print("\nNew network statistics …")
        new_stats = fit_real_stats(synthetic_profiles, synth_graph, political_leaning_key=political_leaning_key)

    if draw_network:
        nodes_to_sample = net_params["draw_network_params"]["sample_n_nodes"]
        group_layout = net_params["draw_network_params"]["group_layout"]
        color_mapping = {
            "far-left": "black", "far-right": "red", "right": "orange", "left": "blue", "center": "pink",
            "non-political": "green"
        }
        for G, l, df_src in [(Real_graph, "Real", "data/clean_data/stance/stance_mistral_real.csv"),
                             (Synthetic_graph, "Synthetic", "data/clean_data/synthetic_posts_10k_final.csv")]:
            if len(G) > 0:
                df = pd.read_csv(df_src)
                title = l + " network by political leaning"
                leanings = df.drop_duplicates(subset=political_leaning_key)[political_leaning_key].tolist()
                df_nodes = df["account_id"].drop_duplicates().astype(str).tolist()
                G_new = G.copy()
                for node in G.nodes():
                    if node not in df_nodes or (l=="Real" and node not in nodes_with_connections):
                        G_new.remove_node(node)
                if nodes_to_sample:
                    sampled_df = stratified_sample(df, n=nodes_to_sample, min_per_label=10)
                    sampled_nodes = sampled_df["account_id"].astype(str).tolist()
                    #sampled_nodes = random.sample(list(G_new.nodes()), nodes_to_sample)
                    G_new = G_new.subgraph(sampled_nodes)
                G = G_new.copy()
                isolated_nodes = list(nx.isolates(G_new))
                G.remove_nodes_from(isolated_nodes)

                order = ["far-right", "right", "center", "non-political", "left", "far-left"]
                leanings_lookup = df.set_index("account_id")[political_leaning_key].to_dict()
                print(len(G.edges))
                plot_network(G, group_layout=group_layout, color_map=color_mapping, leanings_lookup=leanings_lookup,
                            title=title+f", sampled {G.number_of_nodes()} nodes", order=order)

