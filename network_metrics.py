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
        group = group_lookup.get(int(node), "unknown")
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
        g = group_lookup.get(int(node), "unknown")
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
        pos = nx.spring_layout(G, k=1, seed=42)
        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos=pos, node_color=node_colors, node_size=200, font_color="white", font_weight="bold",
                         edge_color="#cccccc", width=8, with_labels=False)
        legend_elements = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
        plt.legend(handles=legend_elements, loc="best", fontsize=11)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def fit_real_stats(real_profiles: pd.DataFrame, real_graph: nx.DiGraph, political_leaning_key) -> dict:
    """
    Returns a dict keyed by political group with:
      - 'out_degrees' : np.array of observed out-degrees (for empirical sampling)
      - 'intra_ratio' : fraction of out-edges that go to same-group users
    """
    political_leanings = real_profiles[political_leaning_key].drop_duplicates().tolist()
    # map user → group
    id_to_group = dict(zip(real_profiles["account_id"], real_profiles[political_leaning_key]))

    stats = {}
    for group in political_leanings:
        members = set(real_profiles.loc[real_profiles[political_leaning_key] == group, "account_id"])

        out_degrees = []
        intra_counts = []
        inter_counts = []

        for user in members:
            if user not in real_graph:
                out_degrees.append(0)
                continue
            followees = list(real_graph.successors(user))
            out_degrees.append(len(followees))
            intra = sum(1 for f in followees if id_to_group.get(f) == group)
            inter = len(followees) - intra
            intra_counts.append(intra)
            inter_counts.append(inter)

        total_out = sum(intra_counts) + sum(inter_counts)
        intra_ratio = sum(intra_counts) / total_out if total_out > 0 else 0.5

        stats[group] = {
            "out_degrees": np.array(out_degrees, dtype=int),
            "intra_ratio": intra_ratio,
        }

        print(f"  [{group:12s}]  median_out_degree={np.median(out_degrees):.1f}"
              f"  mean_out_degree={np.mean(out_degrees):.1f}"
              f"  intra_ratio={intra_ratio:.2f}")

    return stats


def build_candidate_index(raw_connections: list, synth_profiles: pd.DataFrame, threshold: float, political_leaning_key) -> dict:
    """
    For each synthetic user u1 (follower), build a sorted list of
    (score, u2) candidates, filtered by MIN_SCORE_FLOOR.

    Returns: { u1_id : [(score, u2_id), ...] }  sorted descending by score
    """
    synth_ids = set(synth_profiles["account_id"].astype(str))
    id_to_group = dict(zip(synth_profiles["account_id"], synth_profiles[political_leaning_key]))

    # group u2 candidates by u1
    raw_index: dict = defaultdict(list)
    for u1, u2, score in raw_connections:
        u1, u2 = str(u1), str(u2)
        if u1 not in synth_ids or u2 not in synth_ids:
            continue
        if score < threshold:
            continue
        raw_index[u1].append((score, u2))

    # sort descending
    candidate_index = {}
    for u1, pairs in raw_index.items():
        pairs.sort(key=lambda x: x[0], reverse=True)
        candidate_index[u1] = pairs

    return candidate_index, id_to_group

def select_edges_for_user(u1: str, u1_group: str, candidates: list, id_to_group: dict, target_n: int, intra_ratio: float) -> list:
    """
    Split candidates into intra-group and inter-group pools (preserving
    score order within each), then fill the intra and inter quotas greedily.
    Falls back gracefully when a pool is exhausted.
    """
    intra_pool = [(s, u2) for s, u2 in candidates if id_to_group.get(u2) == u1_group]
    inter_pool = [(s, u2) for s, u2 in candidates if id_to_group.get(u2) != u1_group]

    n_intra = round(intra_ratio * target_n)
    n_inter = target_n - n_intra

    # take as many as available; spill overflow into the other pool
    chosen_intra = intra_pool[:n_intra]
    intra_deficit = n_intra - len(chosen_intra)

    chosen_inter = inter_pool[:n_inter + intra_deficit]
    inter_deficit = max(0, n_inter - len(inter_pool))

    # fill inter deficit from remaining intra
    if inter_deficit > 0:
        extra_intra = intra_pool[n_intra: n_intra + inter_deficit]
        chosen_intra = chosen_intra + extra_intra

    chosen = chosen_intra + chosen_inter
    return [(u1, u2) for _, u2 in chosen]


def build_synthetic_edges(synth_profiles: pd.DataFrame, candidate_index: dict, id_to_group: dict, real_stats: dict,
                          political_leaning_key) -> list:
    """
    Main loop: for every synthetic user, sample a target degree from the
    real empirical distribution of their group, then call select_edges_for_user.
    """
    all_edges = []

    for _, row in synth_profiles.iterrows():
        u1 = str(row["account_id"])
        group = row[political_leaning_key]

        group_stats = real_stats[group]
        candidates = candidate_index.get(u1, [])

        if not candidates:
            continue  # no scored pairs for this user

        # sample target degree from empirical distribution of the real group
        target_n = int(np.random.choice(group_stats["out_degrees"]))
        target_n = max(0, target_n)

        if target_n == 0:
            continue

        edges = select_edges_for_user(u1=u1, u1_group=group, candidates=candidates, id_to_group=id_to_group,
                                      target_n=target_n, intra_ratio=group_stats["intra_ratio"])
        all_edges.extend(edges)

    # deduplicate (same pair could appear from both sides in theory)
    all_edges = list(set(all_edges))
    # remove self-loops
    all_edges = [(u, v) for u, v in all_edges if u != v]
    return all_edges


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
    political_leaning_key = "political_leaning"

    nodes_with_connections = set()
    Real_graph = nx.DiGraph()
    Synthetic_graph = nx.DiGraph()

    if real_edges_src:
        with open(real_edges_src, "rb") as f:
            for l in f.readlines():
                e1, e2 = l.split()
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                real_network.append((e1, e2))
                nodes_with_connections.add(e1)
                nodes_with_connections.add(e2)
        Real_graph.add_edges_from(real_network)

    if synthetic_edges_src:
        with open(synthetic_edges_src, "rb") as f:
            el = pickle.load(f)
        synthetic_network = [(int(e[0]), int(e[1])) for e in el]
        Synthetic_graph.add_edges_from(synthetic_network)

    if compute_metrics:
        compute_network_metrics(Real_graph)
        compute_network_metrics(Synthetic_graph)

    if mirror_degree_distribution:
        output_edges_path = net_params["mirroring_node_degrees_params"]["output_edges_path"]
        real_profiles_src = net_params["mirroring_node_degrees_params"]["real_profiles_src"]
        synthetic_profiles_src = net_params["mirroring_node_degrees_params"]["synthetic_profiles_src"]

        real_profiles = pd.read_csv(real_profiles_src)
        synthetic_profiles = pd.read_csv(synthetic_profiles_src).drop_duplicates(subset="account_id")
        with open(synthetic_edges_src, "rb") as f:
            raw_connections = pickle.load(f)

        # ── fit real stats ────────────────────────────────────────────────────────
        print("\nFitting real-network statistics …")
        real_stats = fit_real_stats(real_profiles, Real_graph, political_leaning_key=political_leaning_key)

        # ── build candidate index ─────────────────────────────────────────────────
        print("\nBuilding candidate index …")
        candidate_index, synth_id_to_group = build_candidate_index(raw_connections, synthetic_profiles, threshold=threshold, political_leaning_key=political_leaning_key)

        # ── edge selection ────────────────────────────────────────────────────────
        print("\nSelecting edges …")
        synthetic_edges = build_synthetic_edges(synthetic_profiles, candidate_index, synth_id_to_group, real_stats, political_leaning_key=political_leaning_key)
        print(f"  Synthetic edges created: {len(synthetic_edges)}")

        # ── build synthetic graph ─────────────────────────────────────────────────
        synth_graph = nx.DiGraph()
        synth_graph.add_nodes_from(synthetic_profiles["account_id"])
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
            "non-political": "green", "unknown": "magenta"
        }
        for G, l, df_src in [(Real_graph, "Real", "data/clean_data/stance/stance_mistral_real.csv"),
                             (Synthetic_graph, "Synthetic", "data/clean_data/synthetic_posts.csv")]:
            df = pd.read_csv(df_src)
            title = l + " network by political leaning"
            leanings = df.drop_duplicates(subset=political_leaning_key)[political_leaning_key].tolist()
            df_nodes = df["account_id"].drop_duplicates().tolist()
            G_new = G.copy()
            for node in G.nodes():
                if node not in df_nodes or (l=="Real" and node not in nodes_with_connections):
                    G_new.remove_node(node)
            if nodes_to_sample:
                sampled_df = stratified_sample(df, n=nodes_to_sample, min_per_label=10)
                sampled_nodes = sampled_df["account_id"].tolist()
                #sampled_nodes = random.sample(list(G_new.nodes()), nodes_to_sample)
                G_new = G_new.subgraph(sampled_nodes)
            G = G_new.copy()
            isolated_nodes = list(nx.isolates(G_new))
            G.remove_nodes_from(isolated_nodes)

            order = ["far-right", "right", "center", "non-political", "left", "far-left"]
            leanings_lookup = df.set_index("account_id")[political_leaning_key].to_dict()

            plot_network(G, group_layout=group_layout, color_map=color_mapping, leanings_lookup=leanings_lookup,
                        title=title+f", sampled {G.number_of_nodes()} nodes", order=order)

