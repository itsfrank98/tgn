import numpy as np
import pandas as pd
import os
from tqdm import tqdm

BASE_DIR = "/home/francesco/tgn/data/clean_data" # "/home/francesco/tgn/data/with_synthetic/10k_users"
OUTPUT_DIR = "data/preprocess_outputs"
data_names = ["gab1", "gab_posts_10k_synthetic", "gab_posts_no_zero_snap0"]
data_name = data_names[2]      # ---> fatto preprocessing per il caso baseline
bipartite = False

def preprocess_type_interactions(data_name):
    u_list, i_list, ts_list, label_list, idx_list, type_list, node_features, edge_features = [], [], [], [], [], [], [], []
    node_features_idx = []
    edge_features_idx = []
    if data_name.endswith(".tsv"):
        separator = "\t"
    elif data_name.endswith(".csv"):
        separator = ","
    with open(data_name) as f:
        _ = next(f)
        for idx, line in tqdm(enumerate(f)):
            e = line.strip().replace('"', '').split(separator)
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            label = float(e[3])  # int(e[3])
            if separator == "\t":
                arstr = e[4].split(",")
                feat = np.array([float(x) for x in arstr])
            elif separator == ",":
                feat = np.array([float(x) for x in e[4:-1]])
            interaction_type = e[-1].split(".")[0]

            if interaction_type == '0':
                edge_features_idx.append(len(edge_features))
                node_features_idx.append(-1)
                edge_features.append(feat)
            elif interaction_type == '1':
                node_features_idx.append(len(node_features))
                edge_features_idx.append(-1)
                node_features.append(feat)
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            type_list.append(int(interaction_type))
        df = pd.DataFrame({'u': u_list,
                          'i': i_list,
                          'ts': ts_list,
                          'label': label_list,
                          'type': type_list,
                          'node_idx': node_features_idx,
                          'edge_idx': edge_features_idx,
                          })
        df["global_idx"] = df.index
        return df, np.array(edge_features), np.array(node_features)


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    # Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = os.path.join(BASE_DIR, f"{data_name}.tsv")
    OUT_DF = os.path.join(BASE_DIR, f"ml_{data_name}.tsv")
    OUT_EDGE_FEAT = os.path.join(OUTPUT_DIR, f"ml_{data_name}_edge.npy")
    OUT_NODE_FEAT = os.path.join(OUTPUT_DIR, f"ml_{data_name}_node.npy")

    if data_name == "reddit":
        df, feat = preprocess(PATH)
        new_df = reindex(df, bipartite)

        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])

        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = np.zeros((max_idx + 1, 172))

        new_df.to_csv(OUT_DF)
        np.save(OUT_EDGE_FEAT, feat)
        np.save(OUT_NODE_FEAT, rand_feat)

    else:
        df, edge_features, node_features = preprocess_type_interactions(PATH)
        #df = reindex(df, bipartite)
        np.save(OUT_EDGE_FEAT, edge_features)
        np.save(OUT_NODE_FEAT, node_features)
        df.to_csv(OUT_DF, sep="\t", quoting=3, escapechar="\\", index=False)

run(data_name, bipartite=bipartite)