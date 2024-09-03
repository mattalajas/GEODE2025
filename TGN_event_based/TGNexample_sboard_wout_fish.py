# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score, root_mean_squared_error
from torch.nn import Linear

from torch_geometric.data.temporal import TemporalData
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from torch.utils.tensorboard import SummaryWriter
from TGN_modules import *

import datetime
import pandas as pd
import numpy as np
import pyarrow
import tqdm

summary_writer = 'TGN_new_wout_fish'
event_path = 'GNNthesis/data/Starboard/events.parquet'
data_path = 'GNNthesis/data/Starboard/vessels.csv'
batch_size = 5000
val_ratio = 0.15
test_ratio = 0.15
memory_dim = time_dim = embedding_dim = 100
epochs = 50
lr = 0.0001
l1_reg = 0
verbose = True
self_loop = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

if verbose: writer = SummaryWriter(f'GNNthesis/runs/{summary_writer}_self_{self_loop}')

# Starboard data
data_events = pd.read_parquet(event_path)
data_vessels = pd.read_csv(data_path)

events = ['PORT', 'ECTR']
event_dict = dict(zip(events, range(len(events))))
rev_event_dict = dict(zip(range(len(events)), events))
data_events = data_events[data_events.event_type != 'FISH']
feature_dict = {x['vessel_id']: x['label'] for _, x in data_vessels.iterrows()}

# Initialise inputs 
src = np.zeros((len(data_events)))
dst = np.zeros((len(data_events)))
t = np.zeros((len(data_events)))
# features = (lat_n,lat_s,lon_e,lon_w,ves1,ves2)
features = np.zeros((len(data_events), 4))
y = np.zeros((len(data_events), len(events)))

# convert timesteps
data_events['start_time'] = pd.to_datetime(data_events['start_time'], format='%d/%m/%Y %H:%M')
data_events = data_events.sort_values(by='start_time').reset_index(drop=True)
timesteps = data_events['start_time'].dt.dayofyear

# add fish to events
data_events['event_type'] = data_events['event_type'].fillna('PORT')

prog_bar = tqdm.tqdm(range(len(data_events)))
for ind, data in data_events.iterrows():
    # Define source and dest array
    src[ind] = data['vessel_id']

    if not np.isnan(data['vessel_id2']):
        dst[ind] = data['vessel_id2']
    elif not np.isnan(data['port_id']):
        dst[ind] = data['port_id']

    # Timestamp
    t[ind] = timesteps[ind]

    # Features
    features[ind] = np.array([data['lat_n'], data['lat_s'], data['lon_e'], data['lon_w']])

    # Event types
    event = data['event_type']
    y[ind][event_dict[event]] = 1
    prog_bar.update(1)
    
prog_bar.close()

vals, indexes = np.unique(np.concatenate((src, dst)), return_inverse=True)
src, dst = np.split(indexes, 2)
features = np.nan_to_num(features)

src = torch.Tensor(src).type(torch.long)
dst = torch.Tensor(dst).type(torch.long)
t = torch.Tensor(t).type(torch.long)
features = torch.Tensor(features)
y = torch.Tensor(y)

data = TemporalData(src=src, dst=dst, t=t, msg=features, y=y)
ind_map = {vals[i]: i for i in range(len(vals))}
rev_ind_map = {i: vals[i] for i in range(len(vals))}

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
# dataset = JODIEDataset(path, name='wikipedia')
# data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=val_ratio, test_ratio=val_ratio)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=batch_size,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=batch_size,
    neg_sampling_ratio=1.0,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=batch_size,
    neg_sampling_ratio=1.0,
)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=lr, weight_decay=l1_reg)
criterion_c = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

def train(train_loader):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    prog_bar = tqdm.tqdm(range(len(train_loader)))
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        # y_true = torch.stack((torch.ones(len(batch)).to(device), batch.t), dim=1)
        # loss = criterion_c(pos_out, y_true)

        loss = criterion_c(pos_out, torch.ones_like(pos_out))
        loss += criterion_c(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

        prog_bar.update(1)
    prog_bar.close()
    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, test = False):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    wrong_data = []
    right_data = []

    accs, aps, aucs = [], [], []
    prog_bar = tqdm.tqdm(range(len(loader)))
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_pred = y_pred.view(-1)
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)
        
        if test:
            def get_time(days, year = 2024):
                return datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)

            all_vals = abs(torch.subtract(pos_out.sigmoid().cpu().view(-1), torch.ones(pos_out.size(0))))
            lindices = np.where(all_vals > 0.5)
            windices = np.where(all_vals <= 0.5)

            src = batch.src.cpu().numpy()
            dst = batch.dst.cpu().numpy()
            y = batch.y.cpu().numpy()
            t = np.float64(batch.t.cpu().numpy())

            src = np.vectorize(rev_ind_map.get)(src)
            dst = np.vectorize(rev_ind_map.get)(dst)
            y = np.vectorize(rev_event_dict.get)(np.where(y == 1)[1])
            t = np.vectorize(get_time)(t)

            new_arr = np.vstack((t, src, dst, y)).T
            wrong_data.append(new_arr[lindices])
            right_data.append(new_arr[windices])

        accs.append(root_mean_squared_error(y_true, y_pred))
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        prog_bar.update(1)
    prog_bar.close()

    if test:
        col_vals = ['start_time','vessel_id1', 'vessel_id2', 'event_type']
        wrong_data = np.concatenate(wrong_data)
        right_data = np.concatenate(right_data)

        wrong_df = pd.DataFrame(data = wrong_data,
                          index = list(range(wrong_data.shape[0])),
                          columns = col_vals)
        right_df = pd.DataFrame(data = right_data,
                          index = list(range(right_data.shape[0])),
                          columns = col_vals)
        
        wrong_df.to_csv('GNNthesis/res/wrong_data_fish.csv')
        right_df.to_csv('GNNthesis/res/right_data_fish.csv')
    
    return float(torch.tensor(accs).mean()), float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

for epoch in range(1, epochs+1):
    print(f'Epoch: {epoch:02d}')
    
    loss = train(train_loader)
    val_rmse, val_ap, val_auc = test(val_loader)
    if epoch == 150:
        test_rmse, test_ap, test_auc = test(test_loader, True)
    else:
        test_rmse, test_ap, test_auc = test(test_loader)
    
    print(f'Train Loss: {loss:.4f}')
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

    if verbose:
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('val_rmse', val_rmse, epoch)
        writer.add_scalar('val_ap', val_ap, epoch)
        writer.add_scalar('val_auc', val_auc, epoch)
        writer.add_scalar('test_rmse', test_rmse, epoch)
        writer.add_scalar('test_ap', test_ap, epoch)
        writer.add_scalar('test_auc', test_auc, epoch)

if verbose: writer.close()

##############################################################
# Get some predictions to show
raise
time_vals = np.arange(max(t)+1, max(t)+25, 5)

permut = np.array(np.meshgrid(vals, vals, time_vals)).T.reshape(-1,3)
pred_src, pred_dst, pred_t = np.hsplit(permut, 3)

pred_src = np.vectorize(ind_map.get)(pred_src)
pred_dst = np.vectorize(ind_map.get)(pred_dst)

pred_src = torch.Tensor(pred_src).type(torch.long)
pred_dst = torch.Tensor(pred_dst).type(torch.long)
pred_t = torch.Tensor(pred_t).type(torch.long)

pred_data = TemporalData(src=pred_src, dst=pred_dst, t=pred_t)

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
# dataset = JODIEDataset(path, name='wikipedia')
# data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
pred_data = pred_data.to(device)

batch_size = 200

pred_loader = TemporalDataLoader(
    pred_data,
    batch_size=batch_size
)

final_pred = torch.zeros((pred_src.shape[0], 4)).to(device)

with torch.no_grad():
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    for ind, batch in enumerate(pred_loader):
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        time = batch.t.type(torch.cuda.FloatTensor)
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]], time)
        cur_pred = torch.stack((pos_out.sigmoid().view(-1, 1), batch.src, batch.dst, time), axis = 1)
        final_pred[batch_size*ind:batch_size*(ind+1)] = cur_pred[:,:,0]

final_dataframe = pd.DataFrame(final_pred.detach().cpu().numpy())
final_dataframe = final_dataframe.replace({1:rev_ind_map, 2:rev_ind_map})
final_dataframe = final_dataframe.sort_values(by=[0], ascending=False)
final_dataframe.to_csv('res/first_csv.csv')

