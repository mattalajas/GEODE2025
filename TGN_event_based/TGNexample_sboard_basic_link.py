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

import torch_geometric.transforms as T
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from TGN_modules import *

import networkx as nx
import pandas as pd
import numpy as np
import pyarrow
import tqdm

summary_writer = 'basic_link_pred_new'
event_path = 'GNNthesis/data/Starboard/events.parquet'
data_path = 'GNNthesis/data/Starboard/vessels.csv'
batch_size = 100
val_ratio = 0.1
test_ratio = 0.1
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

# Initialise inputs 
G = nx.Graph()

# Initialise mappings
num_vessels = 0
n_colormap = {}
feature_dict = {x['vessel_id']: x['label'] for _, x in data_vessels.iterrows()}

node_att_map = {node: ind for ind, node in enumerate(set(data_vessels['label']))}
node_att_map['Port'] = len(node_att_map)
node_att_map['NA'] = len(node_att_map)
node_att_map['Fish'] = len(node_att_map)
edge_att_map = {'ECTR': 0, 'FISH': 1, 'DOCK': 2}
eye = np.eye(len(node_att_map))

# Creating graph
prog_bar = tqdm.tqdm(range(len(data_events)))
for ind, data in data_events.iterrows():
    init_vessel = data['vessel_id']
    sec_vessel = data['vessel_id2']
    port = data['port_id']
    event = data['event_type']

    # G.add_node(init_vessel, name = 'vessel', label = feature_dict[init_vessel])
    G.add_node(init_vessel, name = 'vessel')
    n_colormap[init_vessel] = 'blue'
    
    # if not np.isnan(sec_vessel):
    if not np.isnan(sec_vessel):# and sec_vessel in feature_dict:
        if sec_vessel in feature_dict:
            # label = feature_dict[sec_vessel]
            label = 'NA'
            G.add_node(sec_vessel, name = 'vessel', label = label,
                       one_hot = eye[node_att_map[label]])
        else:
            G.add_node(sec_vessel, name = 'vessel', label = 'NA',
                       one_hot = eye[node_att_map['NA']])

        G.add_edge(init_vessel, sec_vessel, event = event, color = 'blue')
        n_colormap[sec_vessel] = 'blue'

    elif not np.isnan(port):
        # G.add_node(port, name = 'port', label = 'Port',
        #            one_hot = eye[node_att_map['Port']])
        G.add_node(port, name = 'port')
        G.add_edge(init_vessel, port, event = 'DOCK', color = 'red')
        n_colormap[port] = 'red'
    
    else:
        if self_loop:
            G.add_edge(init_vessel, init_vessel, event = event, color = 'green')
            n_colormap[0] = 'green'
        else:
            G.add_edge(init_vessel, 0, event = event, color = 'green')
            n_colormap[0] = 'green'
    prog_bar.update(1)
    
prog_bar.close()

G.nodes[0]['one_hot'] = eye[node_att_map['Fish']]

# Create unique mapping
mapping = {node: ind for ind, node in enumerate(G.nodes)}
rev_map = {ind: node for ind, node in enumerate(G.nodes)}

# Compressed graph with new node mappings 
H = nx.relabel_nodes(G, mapping, copy = True)

node_att = nx.get_node_attributes(H, 'one_hot')
edge_att = nx.get_edge_attributes(H, 'event')

x = np.array(list(node_att.values()))
edge_index = np.array(H.edges()).T

x = torch.Tensor(x)
edge_index = torch.Tensor(edge_index).type(torch.long)

data = Data(x=x, edge_index=edge_index)
data = data.to(device)

transform = T.RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True)
train_data, val_data, test_data = transform(data)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
)

memory_dim = data.num_features
hid_channel = 128
out_channel = 64

gnn = BasicGAT(
    in_channels=memory_dim,
    hid_channels=hid_channel,
    out_channels=out_channel
).to(device)

optimizer = torch.optim.Adam(set(gnn.parameters()), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()

print(train_data)

def train(train_data):
    gnn.train()

    torch.manual_seed(12345) 

    optimizer.zero_grad()
    train_data = train_data.to(device)
    x = train_data['x']
    edge_index = train_data['edge_index']

    z = gnn(x, edge_index)

    neg_edge_index = negative_sampling(edge_index=edge_index,
                                        num_nodes=train_data.num_nodes,
                                        num_neg_samples=train_data.edge_label_index.size(1),
                                        method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = gnn.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)

    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(test_data):
    gnn.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    test_data = test_data.to(device)

    x = test_data['x']
    edge_index = test_data['edge_index']

    z = gnn(x, edge_index)
    y_pred = gnn.decode(z, test_data.edge_label_index).view(-1).sigmoid()
    y_pred = y_pred.cpu().numpy()

    y_true = test_data.edge_label.cpu().numpy()

    acc = root_mean_squared_error(y_true, y_pred)
    aps = average_precision_score(y_true, y_pred)
    aucs = roc_auc_score(y_true, y_pred)
    
    return acc, aps, aucs

for epoch in range(1, 151):
    loss = train(train_data)
    print(f'Epoch: {epoch:02d}')
    val_acc, val_ap, val_auc = test(val_data)
    test_acc, test_ap, test_auc = test(test_data)
    print(f'Loss: {loss:.4f}')
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

    if verbose:
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('val_rmse', val_acc, epoch)
        writer.add_scalar('val_ap', val_ap, epoch)
        writer.add_scalar('val_auc', val_auc, epoch)
        writer.add_scalar('test_rmse', test_acc, epoch)
        writer.add_scalar('test_ap', test_ap, epoch)
        writer.add_scalar('test_auc', test_auc, epoch)

writer.close()
