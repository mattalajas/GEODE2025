import numpy as np
import scipy.sparse as sp

import torch
import pandas as pd
import numpy as np
import copy
import scipy.sparse as sparse


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_starboard_data(path, interval=1, size = 183):
    # Starboard data
    events = ['FISH', 'PORT', 'ECTR']
    event_dict = dict(zip(events, range(len(events))))
    data_events = pd.read_csv(f'{path}/events.csv')
    data_vessels = pd.read_csv(f'{path}/vessels.csv')
    feature_dict = {x['vessel_id']: x['label'] for _, x in data_vessels.iterrows()}

    # convert timesteps
    data_events['start_time'] = pd.to_datetime(data_events['start_time'], format='%d/%m/%Y %H:%M')
    data_events = data_events.sort_values(by='start_time').reset_index(drop=True)
    timesteps = data_events['start_time'].dt.day_of_year

    # add fish to events
    data_events['event_type'] = data_events['event_type'].fillna('PORT')
    data_events['ectr_id'] = data_events['ectr_id'].astype('Int64')
    data_events['vessel_id2'] = data_events['vessel_id2'].astype('Int64')
    
    adj_time_list = []
    orig_list = []
    all_feats = []

    ranges = np.arange(timesteps.iloc[0], timesteps.iloc[-1], interval)

    cur_date = ranges[0]
    ind_map = {}
    rev_ind_map = {}

    rows = []
    cols = []
    features = []

    for ind, data in data_events.iterrows():
        # Get event
        event = data['event_type']

        # Append new vessel
        if data['vessel_id'] not in ind_map:
            rev_ind_map[len(ind_map)] = data['vessel_id']
            ind_map[data['vessel_id']] = len(ind_map)
            features.append([0, 0, 0])

        # Create new vessels / port depending on encounter type
        match event:
            case 'ECTR':
                if data['vessel_id2'] not in ind_map:
                    rev_ind_map[len(ind_map)] = data['vessel_id2']
                    ind_map[data['vessel_id2']] = len(ind_map)
                    features.append([0, 0, 0])
            case 'PORT':
                if data['port_id'] not in ind_map:
                    rev_ind_map[len(ind_map)] = data['port_id']
                    ind_map[data['port_id']] = len(ind_map)
                    features.append([0, 0, 0])
            case 'FISH':
                pass
        
        # Create the adj list
        if cur_date >= timesteps[ind]:
            rows.append(ind_map[data['vessel_id']])
            features[ind_map[data['vessel_id']]][event_dict[event]] = 1

            match event:
                case 'ECTR':
                    cols.append(ind_map[data['vessel_id2']])
                    features[ind_map[data['vessel_id2']]][event_dict[event]] = 1
                case 'PORT':
                    cols.append(ind_map[data['port_id']])
                    features[ind_map[data['port_id']]][event_dict[event]] = 1
                case 'FISH':
                    cols.append(ind_map[data['vessel_id']])
                    features[ind_map[data['vessel_id']]][event_dict[event]] = 1
            
            # Change this to end of timestep
            if ind + 1 == len(timesteps):
                all_feats.append(copy.deepcopy(features))

                np_row = np.array(rows)
                np_cols = np.array(cols)
                ones = np.ones_like(np_row)

                mat = sparse.csr_matrix((ones, (np_row, np_cols)), shape=(size, size))
                mat_dense = torch.Tensor(mat.todense())
                orig_list.append(mat_dense)
                adj_time_list.append(copy.deepcopy(mat))

        else:
            all_feats.append(copy.deepcopy(features))
            features = [[0, 0, 0] for _ in range(len(features))]

            np_row = np.array(rows)
            np_cols = np.array(cols)
            ones = np.ones_like(np_row)

            mat = sparse.csr_matrix((ones, (np_row, np_cols)), shape=(size, size))
            mat_dense = torch.Tensor(mat.todense())
            orig_list.append(mat_dense)
            adj_time_list.append(copy.deepcopy(mat))

            rows = []
            cols = []
            cur_date += interval

            rows.append(ind_map[data['vessel_id']])
            features[ind_map[data['vessel_id']]][event_dict[event]] = 1

            match event:
                case 'ECTR':
                    cols.append(ind_map[data['vessel_id2']])
                    features[ind_map[data['vessel_id2']]][event_dict[event]] = 1
                case 'PORT':
                    cols.append(ind_map[data['port_id']])
                    features[ind_map[data['port_id']]][event_dict[event]] = 1
                case 'FISH':
                    cols.append(ind_map[data['vessel_id']])
                    features[ind_map[data['vessel_id']]][event_dict[event]] = 1

    print(adj_time_list)
    return adj_time_list, orig_list