import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import pickle

def preprocess_type_interactions(data_name):
  u_list, i_list, ts_list, label_list, feat_l, idx_list, type_list, node_features, edge_features = [], [], [], [], [], [], [], [], []
  features_list = []
  with open(data_name) as f:
    s = next(f)
    for idx, line in tqdm(enumerate(f)):
      e = line.strip().replace('"', '').split(',')
      u = int(e[1])
      i = int(e[2])

      ts = float(e[3])
      label = float(e[4])  # int(e[3])
      feat = np.array([float(x) for x in e[5:-1]])
      interaction_type = e[-1]

      if interaction_type == '0':
          edge_features.append(feat)
      elif interaction_type == '1':
          node_features.append(feat)

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)
      feat_l.append(feat)
      type_list.append(int(interaction_type))
      features_list.append(feat)
  return (pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'type': type_list,
                       'idx': idx_list,
                       'features': features_list}))
          #, np.array(edge_features), np.array(node_features))

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

BASE_DIR = "/home/francesco/tgn/data"
def run(data_name, bipartite=True):
  #Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = os.path.join(BASE_DIR, f"{data_name}_new.csv")
  OUT_DF = os.path.join(BASE_DIR, f"ml_{data_name}_new_repr.csv")
  OUT_FEAT = os.path.join(BASE_DIR, f"ml_{data_name}.npy")
  OUT_NODE_FEAT = os.path.join(BASE_DIR, f"ml_{data_name}_node.npy")

  if data_name == "reddit":
    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

  else:
    df = preprocess_type_interactions(PATH)     #, edge_feat, node_feat
    new_df = reindex(df, bipartite)
    new_df.to_csv(OUT_DF)

    """empty = np.zeros(edge_feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, edge_feat])

    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, node_feat)"""

"""parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()"""

data_name = "gab"
bipartite = False
run(data_name, bipartite=bipartite)
