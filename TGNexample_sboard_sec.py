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

import StarDataset
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
verbose = False

if verbose: writer = SummaryWriter('runs/TGN_with_time_pred')

# Starboard data
events = ['FISH', 'PORT', 'ECTR']
event_dict = dict(zip(events, range(len(events))))
data_events = pd.read_csv('data/Starboard/events.csv')
data_vessels = pd.read_csv('data/Starboard/vessels.csv')
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

for ind, data in data_events.iterrows():
    # Define source and dest array
    src[ind] = data['vessel_id']

    if not np.isnan(data['vessel_id2']):
        dst[ind] = data['vessel_id2']
    elif not np.isnan(data['port_id']):
        dst[ind] = data['port_id']
    else:
        dst[ind] = 0

    # Timestamp
    t[ind] = timesteps[ind]

    # Features
    features[ind] = np.array([data['lat_n'], data['lat_s'], data['lon_e'], data['lon_w']])

    # Event types
    event = data['event_type']
    y[ind][event_dict[event]] = 1

vals, indexes = np.unique(np.concatenate((src, dst)), return_inverse=True)
src, dst = np.split(indexes, 2)
features = np.nan_to_num(features)

src = torch.Tensor(src).type(torch.long)
dst = torch.Tensor(dst).type(torch.long)
t = torch.Tensor(t).type(torch.long)
features = torch.Tensor(features)
y = torch.Tensor(y)

data = TemporalData(src=src, dst=dst, t=t, msg=features, y=y)

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
# dataset = JODIEDataset(path, name='wikipedia')
# data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=100,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=100,
    neg_sampling_ratio=1.0,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=100,
    neg_sampling_ratio=1.0,
)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final1 = Linear(in_channels, 128)
        self.lin_final2 = Linear(128, 2)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        h = self.lin_final1(h).relu()
        return self.lin_final2(h)


memory_dim = time_dim = embedding_dim = 100

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
    | set(link_pred.parameters()), lr=0.0001)

criterion_c = torch.nn.BCEWithLogitsLoss()
criterion_r = torch.nn.MSELoss()
window = 366

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        pos_out_full = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out_full = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        pos_out = pos_out_full[:, 0]
        neg_out = neg_out_full[:, 0]

        reg_out = pos_out_full[:, 1]
        cur_t = batch.t.view(-1) / window

        # y_true = torch.stack((torch.ones(len(batch)).to(device), batch.t), dim=1)
        # loss = criterion_c(pos_out, y_true)

        c_loss = criterion_c(pos_out, torch.ones_like(pos_out))
        c_loss += criterion_c(neg_out, torch.zeros_like(neg_out))
        r_loss = criterion_r(reg_out, cur_t)
        loss = c_loss + r_loss

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs, rmse = [], [], []
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        pos_out_full = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out_full = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        pos_out = pos_out_full[:, 0]
        neg_out = neg_out_full[:, 0]

        reg_out = pos_out_full[:, 1].sigmoid().cpu()
        cur_t = batch.t.view(-1) / window
        cur_t = cur_t.sigmoid().cpu()

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))
        rmse.append(root_mean_squared_error(reg_out, cur_t))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean()), float(torch.tensor(rmse).mean())

for epoch in range(1, 151):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc, val_rmse = test(val_loader)
    test_ap, test_auc, test_rmse = test(test_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}, Val RMSE: {val_rmse:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Test RMSE: {test_rmse:.4f}')

    if verbose:
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('val_ap', val_ap, epoch)
        writer.add_scalar('val_auc', val_auc, epoch)
        writer.add_scalar('val_rmse', val_rmse, epoch)
        writer.add_scalar('test_ap', test_ap, epoch)
        writer.add_scalar('test_auc', test_auc, epoch)
        writer.add_scalar('test_rmse', test_rmse, epoch)

if verbose: writer.close()

##############################################################
# Get predictions
ind_map = {vals[i]: i for i in range(len(vals))}
rev_ind_map = {i: vals[i] for i in range(len(vals))}

permut = np.array(np.meshgrid(vals, vals)).T.reshape(-1,2)
pred_src, pred_dst = np.hsplit(permut, 2)

pred_src = np.vectorize(ind_map.get)(pred_src)
pred_dst = np.vectorize(ind_map.get)(pred_dst)

pred_src = torch.Tensor(pred_src).type(torch.long)
pred_dst = torch.Tensor(pred_dst).type(torch.long)
pred_t = torch.ones_like(pred_dst).type(torch.long)

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
        pos_out_full = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        pos_out = pos_out_full[:, :, 0]

        reg_out = pos_out_full[:, :, 1].sigmoid() * window

        cur_pred = torch.stack((pos_out.sigmoid().view(-1, 1), batch.src, batch.dst, reg_out), axis = 1)
        final_pred[batch_size*ind:batch_size*(ind+1)] = cur_pred[:,:,0]

final_dataframe = pd.DataFrame(final_pred.detach().cpu().numpy())
final_dataframe = final_dataframe.replace({1:rev_ind_map, 2:rev_ind_map})
final_dataframe = final_dataframe.sort_values(by=[0], ascending=False)
final_dataframe.to_csv('res/sec_csv.csv')

