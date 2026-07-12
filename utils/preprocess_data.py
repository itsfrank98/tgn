import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import yaml

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


def main(base_dir, output_dir,data_name, ext, bipartite=True):
    # Path("data/").mkdir(parents=True, exist_ok=True)
    path = os.path.join(base_dir, f"{data_name}.{ext}")
    out_df = os.path.join(output_dir, f"ml_{data_name}.{ext}")
    out_edge_feat = os.path.join(output_dir, f"ml_{data_name}_edge.npy")
    out_node_feat = os.path.join(output_dir, f"ml_{data_name}_node.npy")

    df, edge_features, node_features = preprocess_type_interactions(path)
    #df = reindex(df, bipartite)
    np.save(out_edge_feat, edge_features)
    np.save(out_node_feat, node_features)
    df.to_csv(out_df, sep="\t", quoting=3, escapechar="\\", index=False)


if __name__ == "__main__":
    with open("../parameters.yaml", 'r') as params_file:
        args = yaml.safe_load(params_file)

    data_name = args["data"]
    consider_synthetic = args["consider_synthetic"]
    args_prep = args["preprocessing"]
    embedding_type = args_prep["embedding_type"]
    data_name += f"_{embedding_type}"
    if consider_synthetic > 0:
        data_name += f"_with_synthetic_{consider_synthetic}"
    main(base_dir=args_prep["base_dir"], output_dir=args_prep["output_dir"], data_name=data_name,
         bipartite=args_prep["bipartite"], ext=args["ext"])