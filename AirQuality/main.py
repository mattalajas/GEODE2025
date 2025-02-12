import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal

from models import TemporalGNN, LSTMcell, train_AT3GCN, train_LSTM
from utils import AirQualityDataset, BeijingAirQualityDataset, plot_features

def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

    return model


def main():
    ### SETUP ARGS
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--cuda", type=int, default=0, help='Cuda ID')
    parser.add_argument("--model", type=str, default="a3tgcn", choices=["a3tgcn", "lstm"], help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hid_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--seed", type=int, default=None, help="Torch seed")

    parser.add_argument("--verbose", type=bool, default=False, help="Tensorboard writer")
    parser.add_argument("--save_path", type=str, default="AirQuality/models/", help="Path to save trained model")
    parser.add_argument("--features", type=str, default="0,1,2,3", help="Features to use")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "max", "min"], help="Aggregate features")
    parser.add_argument("--location", type=str, default="Auckland", choices=["Auckland", "Christchurch"], help="Location of data")
    parser.add_argument("--time_range", type=str, default="2022-04-01,2022-12-01", help='Time range of dataset')
    parser.add_argument("--interval", type=str, default="hour", help='Granularity of data')
    parser.add_argument("--eq_weights", type=bool, default=False, help='Equal edge weightings')
    parser.add_argument("--start_pred", type=int, default=12, help="When to start prediction")
    parser.add_argument("--train_ratio", type=int, default=0.8, help="Train-test ratio")
    parser.add_argument("--dataset", type=str, default='niwa', choices=['niwa', 'beijing'], help='Dataset')
    parser.add_argument("--temp_data_dir", type=str, default=None, help="Call if pickle file is available")
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
    writer = None
    save_name = f'model_{args.model}_dataset_{args.dataset}_lr_{args.lr}_hid_{args.hid_size}_agg_{args.aggregate}_location_{args.location}_features_{args.features}'
    if args.verbose:
        writer = SummaryWriter(f'AirQuality/save_data/{save_name}')
    
    time_range = args.time_range.split(',')

    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    ### SETUP DATABASE
    fin_cols = []
    if args.dataset == 'niwa':
        COLS = ['pm10ConcNumIndividual.value', 'pm1ConcNumIndividual.value',
                'pm2_5ConcNumIndividual.value', 'relHumidInternalIndividual.value']
        AUCKLAND = {
            'df' :      pd.DataFrame({
                        'locationLatitude': [-36.844079, -36.844113, -36.711932, -36.898491, -36.906652, -36.876728],
                        'locationLongitude': [174.762123, 174.761371, 174.740808, 174.591428, 174.633079, 174.703081]}), 
            'timezone': 'Pacific/Auckland'}
        
        for i in args.features.split(','):
            fin_cols.append(COLS[int(i)])

        features = {feat:args.aggregate for feat in fin_cols}

        if args.location == 'Auckland':
            lat_long_vals = AUCKLAND["df"]
            time_zone = AUCKLAND['timezone']

        niwa_loader = AirQualityDataset(args.data_dir, features, lat_long_vals, 
                                        time_range, time_zone, args.interval,
                                        args.eq_weights)
        fin_dataset = niwa_loader.get_dataset(args.start_pred)

    elif args.dataset == 'beijing':
        COLS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
        BEIJING = {
            'df' :      pd.DataFrame({
                        'locationLatitude': [39.982, 40.217, 40.292, 39.939, 39.929, 39.914, 40.328, 39.937, 40.127, 39.886, 39.987, 39.878],
                        'locationLongitude': [116.397, 116.23, 116.22, 116.483, 116.339, 116.184, 116.628, 116.461, 116.655, 116.407, 116.287, 116.352]})}

        lat_long_vals = BEIJING["df"]

        for i in args.features.split(','):
            fin_cols.append(COLS[int(i)])
        
        features = {feat:args.aggregate for feat in fin_cols}

        beijing_loader = BeijingAirQualityDataset(args.data_dir, features=features, 
                                                   t_range=time_range, interval=args.interval,
                                                   eq_weights=args.eq_weights)
    
        if args.temp_data_dir:
            with open(args.temp_data_dir, 'rb') as handle:
                b = pickle.load(handle)

            fin_dataset = beijing_loader.get_dataset(args.start_pred, b)
        else:
            fin_dataset = beijing_loader.get_dataset(args.start_pred)

    # Train and test set
    train_dataset, test_dataset = temporal_signal_split(fin_dataset, train_ratio=args.train_ratio)
    train_input = np.array(train_dataset.features) # (195, 6, 4, 12)
    train_target = np.array(train_dataset.targets) # (195, 6, 4, 12)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (B, N*F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N*F, T)

    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_input = np.array(test_dataset.features) # (49, 6, 4, 12)
    test_target = np.array(test_dataset.targets) # (49, 6, 4, 12)
    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  

    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f'Train set: {train_input.shape}')
    print(f'Test set: {test_input.shape}')
    
    ### SETUP MODEL

    if args.model == "a3tgcn":
        model = TemporalGNN(node_features=len(fin_cols), periods=24-args.start_pred,
                            out_size=(24-args.start_pred)*len(fin_cols), batch_size=args.batch_size,
                            device=DEVICE).to(DEVICE)
    elif args.model == 'lstm':
        input_size = len(lat_long_vals.index)*len(fin_cols)
        model = LSTMcell(input_size, args.hid_size, input_size).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    #--------------------------------------------------
    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    
    if args.model == 'a3tgcn':
        for snapshot in train_dataset:
            static_edge_index = snapshot.edge_index.to(DEVICE)
            static_edge_weight = snapshot.edge_attr.to(DEVICE)
            break  

        trained_model, all_y_true, all_y_hat = train_AT3GCN(model, loss_fn, optimizer, args.epochs,
                                               train_loader, test_loader, static_edge_index,
                                               static_edge_weight, writer)
    elif args.model == 'lstm':
        trained_model, all_y_true, all_y_hat = train_LSTM(model, loss_fn, optimizer, args.epochs, DEVICE,
                                               train_loader, test_loader, args.hid_size, 
                                               args.start_pred, writer)

    if args.verbose:
        save_path = f"{args.save_path}/{save_name}.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved to {args.save_path}")

        for feat in range(len(fin_cols)):
            plot_features(features, all_y_hat, all_y_true, feat, save_name)

if __name__ == "__main__":
    main()
