import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle

def preprocess(data_name):
  u_list, i_list, ts_list, label_list, feat_l, idx_list, type_list, node_features, edge_features = [], [], [], [], [], [], [], [], []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().replace('"', '').split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])
      feat = np.array([float(x) for x in e[4:-1]])
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
      type_list.append(interaction_type)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'type': type_list,
                       'idx': idx_list}), np.array(edge_features), np.array(node_features)


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
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = '../data/{}.csv'.format(data_name)
  OUT_DF = '../data/ml_{}.csv'.format(data_name)
  OUT_FEAT = '../data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = '../data/ml_{}_node.npy'.format(data_name)
  IN_FEAT = '../data/{}_tensor.npy'.format(data_name)
  src_node_features = np.load(IN_FEAT)

  df, edge_feat, node_feat = preprocess(PATH)
  # TODO ACCERTARSI CHE NELLE EDGE FEATURES CI SIANO SOLO QUELLE DEGLI EDGE E NON ANCHE QUELLE NODE WISE
  new_df = reindex(df, bipartite)

  empty = np.zeros(edge_feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, edge_feat])

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, node_feat)

"""parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()"""

data_name = "gab"
bipartite = False
#run(data_name, bipartite=bipartite)
