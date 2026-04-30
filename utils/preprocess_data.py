###
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import pickle
BASE_DIR = "/home/francesco/tgn/data/clean_data"
OUTPUT_DIR = "/home/francesco/tgn/data/preprocess_outputs"

data_name = "gab_post_follow_comment_nodeft"      # ---> fatto preprocessing per il caso baseline, quello in cui si fa lo shuffling dei dati
bipartite = False

def preprocess_type_interactions(data_name):
    u_list, i_list, ts_list, label_list, idx_list, type_list, node_features, target_types = [], [], [], [], [], [], [], []
    edge_features_post, edge_features_follow, edge_features_comment = [], [], []
    node_features_idx, edge_features_post_idx,  edge_features_follow_idx, edge_features_comment_idx = [], [], [], []
    with open(data_name) as f:
        s = next(f)
        for idx, line in tqdm(enumerate(f)):
            e = line.strip().replace('"', '').split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            interaction_type = int(e[3])
            label = float(e[4])  # int(e[3])
            feat = np.array([float(x) for x in e[5:]])

            if interaction_type == 0:     # FOLLOW
                edge_features_follow_idx.append(len(edge_features_follow))
                edge_features_follow.append(feat)

                node_features_idx.append(-1)
                edge_features_post_idx.append(-1)
                edge_features_comment_idx.append(-1)
                target_types.append(1)
            elif interaction_type == 1:   # NODE FEATURE UPDATE BY AGGREGATING THE USER'S POSTS
                node_features_idx.append(len(node_features))
                node_features.append(feat)

                edge_features_follow_idx.append(-1)
                edge_features_post_idx.append(-1)
                edge_features_comment_idx.append(-1)
                target_types.append(-1)
            elif interaction_type == 2:       # COMMENTED
                edge_features_comment_idx.append(len(edge_features_comment))
                edge_features_comment.append(feat)

                node_features_idx.append(-1)
                edge_features_follow_idx.append(-1)
                edge_features_post_idx.append(-1)
                target_types.append(0)
            elif interaction_type == 3:       # POSTED
                edge_features_post_idx.append(len(edge_features_post))
                edge_features_post.append(feat)

                node_features_idx.append(-1)
                edge_features_follow_idx.append(-1)
                edge_features_comment_idx.append(-1)
                target_types.append(0)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            #idx_list.append(idx)
            #feat_l.append(feat)
            type_list.append(int(interaction_type))
        df = pd.DataFrame({'s': u_list,
                           't': i_list,
                           'ts': ts_list,
                           'label': label_list,
                           'type': type_list,
                           'node_idx': node_features_idx,
                           # 'node_features': node_features,
                           'comment_edge_idx': edge_features_comment_idx,
                           'follow_edg_idx': edge_features_follow_idx,
                           'post_edg_idx': edge_features_post_idx,
                           'target_type': target_types,
                           # 'edge_features': edge_features,
                          })
        df["global_idx"] = df.index
        return df, np.array(edge_features_follow), np.array(edge_features_comment), np.array(edge_features_post), np.array(node_features)


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
    PATH = os.path.join(BASE_DIR, f"{data_name}.csv")
    OUT_DF = os.path.join(OUTPUT_DIR, f"ml_{data_name}.csv")
    OUT_EDGE_FEAT_FOLLOW = os.path.join(OUTPUT_DIR, f"ml_gab_edge_follow.npy")
    OUT_EDGE_FEAT_POST = os.path.join(OUTPUT_DIR, f"ml_gab_edge_post.npy")
    OUT_EDGE_FEAT_COMMENT = os.path.join(OUTPUT_DIR, f"ml_gab_edge_comment.npy")
    OUT_NODE_FEAT = os.path.join(OUTPUT_DIR, f"ml_gab_node.npy")

    if data_name == "reddit":
        df, feat = preprocess(PATH)
        new_df = reindex(df, bipartite)

        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])

        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = np.zeros((max_idx + 1, 172))

        new_df.to_csv(OUT_DF, index=False)
        np.save(OUT_EDGE_FEAT_POST, feat)
        np.save(OUT_NODE_FEAT, rand_feat)

    else:
        df, edge_features_follow, edge_features_comment, edge_features_post, node_features = preprocess_type_interactions(PATH)
        #df = reindex(df, bipartite)
        np.save(OUT_EDGE_FEAT_FOLLOW, edge_features_follow)
        np.save(OUT_EDGE_FEAT_POST, edge_features_post)
        np.save(OUT_EDGE_FEAT_COMMENT, edge_features_comment)
        np.save(OUT_NODE_FEAT, node_features)
        df.to_csv(OUT_DF, index=False)

run(data_name, bipartite=bipartite)
