import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch_geometric_temporal.nn.recurrent import A3TGCN2

from haversine import haversine

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, out_size, batch_size, device):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=512, periods=periods, batch_size=batch_size, device=device) # node_features=4, periods=12
        # Equals single-shot prediction
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, out_size)

    def forward(self, x, edge_index, edge_weights=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        if edge_weights is not None:
            h = self.tgnn(x, edge_index, edge_weights) # x [b, 6, 4, 12]  returns h [b, 6, 12*4]
        else:
            h = self.tgnn(x, edge_index)
        h = F.relu(self.linear1(h)) 
        h = F.relu(self.linear2(h)) 
        h = self.linear3(h)
        return h
    
def train_AT3GCN(model, loss_fn, optimizer, epochs, train_loader, test_loader, static_edge_index, static_edge_weight, writer):
    model.train()

    tbar = trange(epochs, desc="Training iter")
    for epoch in tbar:
        loss_list = []
        for encoder_inputs, labels in train_loader:
            # print(encoder_inputs, static_edge_index.shape)
            y_true = torch.flatten(labels, start_dim=2) # (B, N, F*T)
            y_hat = model(encoder_inputs, static_edge_index, static_edge_weight)         # Get model predictions
            loss = torch.sqrt(loss_fn(y_hat, y_true)) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
        tbar.set_description("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))

        if writer is not None:
            writer.add_scalar('Train RMSE', sum(loss_list)/len(loss_list), epoch)

    model.eval()
    # Store for analysis
    total_loss = []
    all_y_true = []
    all_y_hat = []
    
    for encoder_inputs, labels in test_loader:
        # Get model predictions
        y_true_t = torch.flatten(labels, start_dim=2)
        y_hat_t = model(encoder_inputs, static_edge_index, static_edge_weight)
        # Mean squared error
        loss = torch.sqrt(loss_fn(y_hat_t, y_true_t))
        total_loss.append(loss.item())

        all_y_true.append(y_true_t)
        all_y_hat.append(y_hat_t)
        
    print("Test MSE: {:.4f}".format(sum(total_loss)/len(total_loss)))
    
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_hat = torch.cat(all_y_hat, dim=0)

    all_y_true = torch.unflatten(all_y_true, 2, (labels.shape[2], labels.shape[3]))
    all_y_hat = torch.unflatten(all_y_hat, 2, (labels.shape[2], labels.shape[3]))

    return model, all_y_true, all_y_hat

class LSTMcell(torch.nn.Module):
    def __init__(self, initial_size, fin_emb, out_size, dropout = 0):
        super().__init__()
        # Check input size
        self.lstm1 = torch.nn.LSTMCell(initial_size, fin_emb)

        self.linear1 = torch.nn.Linear(fin_emb, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, out_size)

    def forward(self, x, h0, c0):
        # Feeds concatenated vector to LSTM alongside hidden layer and cell state
        hx, cx = self.lstm1(x, (h0, c0))

        h = F.relu(self.linear1(hx)) 
        h = F.relu(self.linear2(h)) 
        h = self.linear3(h)
        return h, hx, cx 
    
def train_LSTM(model, loss_fn, optimizer, epochs, device, train_loader, test_loader, hid_size, start_pred, writer):
    model.train()

    tbar = trange(epochs, desc="Training iter")
    for epoch in tbar:
        loss_list = []
        for encoder_inputs, o_labels in train_loader:
            print(o_labels)
            raise
            encoder_inputs = torch.flatten(encoder_inputs, start_dim=1, end_dim=2).permute(0, 2, 1) # (B, T, N*F)
            labels = torch.flatten(o_labels, start_dim=1, end_dim=2).permute(0, 2, 1) # (B, T, N*F)

            h0 = torch.ones((labels.shape[0], hid_size)).to(device)
            h0 = torch.nn.init.xavier_uniform_(h0)
            c0 = torch.ones((labels.shape[0], hid_size)).to(device)
            c0 = torch.nn.init.xavier_uniform_(c0)

            losses = torch.empty(0).to(device)

            full_seq = torch.cat([encoder_inputs, labels], dim=1).to(device)

            for seq in range(23):
                if seq < start_pred-1:
                    seq_cur = full_seq[:, seq, :]
                else:
                    seq_cur = seq_t.detach()

                seq_t, h_t, c_t, = model(seq_cur, h0, c0)

                seq_loss = torch.sqrt(loss_fn(seq_t, full_seq[:, seq+1, :]))
                losses = torch.cat((losses, seq_loss.reshape(1)))

                h0 = h_t.detach()
                c0 = c_t.detach()
            
            loss = torch.mean(losses)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
        tbar.set_description("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))

        if writer is not None:
            writer.add_scalar('Train RMSE', sum(loss_list)/len(loss_list), epoch)

    model.eval()
    # Store for analysis
    total_loss = []
    all_y_true = []
    all_y_hat = []
    for encoder_inputs, o_labels in test_loader:
        encoder_inputs = torch.flatten(encoder_inputs, start_dim=1, end_dim=2).permute(0, 2, 1) # (B, T, N*F)
        labels = torch.flatten(o_labels, start_dim=1, end_dim=2).permute(0, 2, 1) # (B, T, N*F)

        pred_seq = []

        h0 = torch.ones((labels.shape[0], hid_size)).to(device)
        h0 = torch.nn.init.xavier_uniform_(h0)
        c0 = torch.ones((labels.shape[0], hid_size)).to(device)
        c0 = torch.nn.init.xavier_uniform_(c0)

        losses = torch.empty(0).to(device)

        full_seq = torch.cat([encoder_inputs, labels], dim=1).to(device)

        for seq in range(23):
            if seq < start_pred-1:
                seq_cur = full_seq[:, seq, :]
            else:
                pred_seq.append(seq_t)
                seq_cur = seq_t

            seq_t, h_t, c_t, = model(seq_cur, h0, c0)

            seq_loss = torch.sqrt(loss_fn(seq_t, full_seq[:, seq+1, :]))
            losses = torch.cat((losses, seq_loss.reshape(1)))

            h0 = h_t
            c0 = c_t

        # Get model predictions
        loss = torch.mean(losses[start_pred:])
        total_loss.append(loss.item())

        all_y_hat.append(torch.stack(pred_seq))
        all_y_true.append(full_seq[:, start_pred:, :])
    
    print("Test MSE: {:.4f}".format(sum(total_loss)/len(total_loss)))

    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_hat = torch.cat(all_y_hat, dim=1)
    print(all_y_hat.shape, all_y_true.shape)

    all_y_true = torch.unflatten(all_y_true, 2, (o_labels.shape[1], o_labels.shape[2])).permute(0, 2, 3, 1)
    all_y_hat = torch.unflatten(all_y_hat, 2, (o_labels.shape[1], o_labels.shape[2])).permute(1, 2, 3, 0)    
    print(all_y_hat.shape, all_y_true.shape)

    return model, all_y_true, all_y_hat

    