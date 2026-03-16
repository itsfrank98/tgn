import os.path

import numpy as np
from itertools import combinations
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import networkx as nx
import pickle
import matplotlib.pyplot as plt

def save_to_pickle(name, c):
  with open(name, 'wb') as f:
    pickle.dump(c, f)


def load_from_pickle(name):
  with open(name, 'rb') as f:
    return pickle.load(f)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def node_neighbor_similarity_distribution(G, attr="features"):
    sims = []
    for node in tqdm(G.nodes()):
        x_u = np.array(G.nodes[node][attr])
        neighbors = list(G.successors(node))
        if len(neighbors) == 0:
            continue
        neigh_feats = np.array([G.nodes[v][attr] for v in neighbors])
        h_u = neigh_feats.mean(axis=0)
        sims.append(cosine(x_u, h_u))
    return np.array(sims)


def compare_networks(G1, G2, attr="features"):
    dist1 = node_neighbor_similarity_distribution(G1, attr)
    dist2 = node_neighbor_similarity_distribution(G2, attr)
    W = wasserstein_distance(dist1, dist2)
    return dist1, dist2, W

if __name__ == "__main__":
    edge_src_real = "data/clean_data/social_network.edg"
    edge_src_synthetic = "connections_2.pkl"
    features_src_real = "data/clean_data/bert_features_real_users.pkl"
    features_src_synthetic = "data/clean_data/bert_features_synthetic_users.pkl"
    threshold = .65
    edges_real = []

    with open(edge_src_real, "rb") as f:
        for l in f.readlines():
            e1, e2 = l.split()
            e1 = int(e1.strip())
            e2 = int(e2.strip())
            edges_real.append((e1, e2))
    el = load_from_pickle(edge_src_synthetic)
    edges_synthetic = [(int(e[0]), int(e[1])) for e in el if e[2] > threshold]
    feats_real = load_from_pickle(features_src_real)
    feats_synthetic = load_from_pickle(features_src_synthetic)
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    G1.add_edges_from(edges_real)
    G2.add_edges_from(edges_synthetic)

    for t in [(G1, feats_real), (G2, feats_synthetic)]:
        fts = t[1]
        remove = []
        for n in t[0].nodes():
            try:
                t[0].nodes[n]["features"] = fts[n]
            except KeyError:
                remove.append(n)
        t[0].remove_nodes_from(remove)
    if not os.path.exists("dist1.pkl"):
        dist1, dist2, W = compare_networks(G1, G2)
        save_to_pickle("dist1.pkl", dist1)
        save_to_pickle("dist2.pkl", dist2)
        print("distance: ", W)
    else:
        dist1 = load_from_pickle("dist1.pkl")
        dist2 = load_from_pickle("dist2.pkl")
    plt.hist(dist1, bins=30, alpha=0.5, label="Real")
    #plt.hist(dist2, bins=30, alpha=0.5, label="Synthetic")
    plt.legend()
    plt.xlabel("Neighbor similarity")
    plt.ylabel("Frequency")
    plt.show()