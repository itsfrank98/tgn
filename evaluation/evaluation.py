import math
import numpy as np
import torch
from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]
            node_idxs_batch = data.node_idxs[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            size = len(sources_batch)

            if hasattr(data, 'interaction_types'):
                interaction_types_batch = data.interaction_types[s_idx:e_idx]
                for j in range(size):
                    t = interaction_types_batch[j]
                    if t == 1:
                        node = sources_batch[j]
                        feats = model.node_raw_features[node_idxs_batch][j]
                        timestamp = timestamps_batch[j]
                        model.process_node_wise_event(node=node, timestamp=timestamp, new_feature=feats)
                edge_mask = (interaction_types_batch == 0)
                if edge_mask.sum() > 0:  # only if there are edge events in batch
                    # Only sample negatives and compute loss for real edges
                    sources_batch = sources_batch[edge_mask]
                    destinations_batch = destinations_batch[edge_mask]
                    timestamps_batch = timestamps_batch[edge_mask]
                    edge_idxs_batch = edge_idxs_batch[edge_mask]
                    size = len(sources_batch)
                else:
                    size = 0
            if size > 0:
                max_valid_idx = model.edge_raw_features.shape[0] - 1
                _, negative_samples = negative_edge_sampler.sample(size)

                pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, n_neighbors)

                pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
                true_label = np.concatenate([np.ones(size), np.zeros(size)])

                val_ap.append(average_precision_score(true_label, pred_score))
                val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def predict_connections(model, data, t_query, prob_threshold=0.5, batch_size=200):
    # Step 1: Reset state
    model.latest_node_features.copy_(model.node_raw_features)
    if model.use_memory:
        model.memory.reset_memory_states()

    # Step 2: Process node-wise events
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    predicted_edges = []
    for k in range(num_test_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

        sources_batch = data.sources[s_idx:e_idx]
        node_idxs_batch = data.node_idxs[s_idx:e_idx]
        timestamps_batch = data.timestamps[s_idx:e_idx]
        size = len(sources_batch)

        if hasattr(data, 'interaction_types'):
            interaction_types_batch = data.interaction_types[s_idx:e_idx]
            for j in range(size):
                t = interaction_types_batch[j]
                if t == 1:
                    node = sources_batch[j]
                    feats = model.node_raw_features[node_idxs_batch][j]
                    timestamp = timestamps_batch[j]
                    model.process_node_wise_event(node=node, timestamp=timestamp, new_feature=feats)
        """for event in sorted_node_wise_events:
            feat_vec = torch.from_numpy(event.feat_vec).float().to(model.device)
            model.process_node_wise_event(event.node_id, event.ts, feat_vec)"""

        # Step 3: Compute embeddings
        model.eval()
        with torch.no_grad():
            nodes_tensor = torch.tensor(sources_batch).to(model.device)
            ts_tensor = torch.full_like(nodes_tensor, t_query, dtype=torch.float)
            embeddings = model.embedding_module.compute_embedding(
                memory=model.memory,
                source_nodes=sources_batch,
                timestamps=timestamps_batch,
                n_neighbors=0,  # no neighbors for isolated
                n_layers=model.n_layers,
            )  # shape [len(nodes), embed_dim]

        # Step 4: Predict pairs
        for i, j in combinations(range(len(nodes_tensor)), 2):
            z_i = embeddings[i].unsqueeze(0)
            z_j = embeddings[j].unsqueeze(0)
            prob = model.affinity_score(z_i, z_j).sigmoid().item()
            if prob > prob_threshold:
                predicted_edges.append((sources_batch[i], sources_batch[j], prob))

    # Sort by confidence
        predicted_edges.sort(key=lambda x: x[2], reverse=True)
    return predicted_edges

def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
