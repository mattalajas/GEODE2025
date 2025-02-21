import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange

from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from haversine import haversine

class BeijingAirQualityDataset():
    def __init__(self, path, features=None, lat_long_vals=None, t_range=None, time_zone=None,  interval='hour', eq_weights=True):
        self.eq_weights = eq_weights
        self.interval = interval
        
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], unit="ns", utc=True)
        df['locationLatitude'] = df['locationLatitude'].round(6)
        df['locationLongitude'] = df['locationLongitude'].round(6)
        cols_to_keep = ['time', 'hour', 'locationLatitude', 'locationLongitude'] + list(features.keys())

        if features:
            df = df[cols_to_keep]
        if time_zone:
            df.time = df.time.dt.tz_convert(time_zone)
        if t_range:
            df = df[(df['time'] > pd.to_datetime(t_range[0],unit="ns", utc=True)) 
                    & (df['time'] < pd.to_datetime(t_range[1],unit="ns", utc=True))]
        if lat_long_vals is not None:
            df = df.merge(lat_long_vals, on=['locationLatitude', 'locationLongitude'])

        fin_df = df.groupby(['locationLatitude', 'locationLongitude', pd.Grouper(key='time', freq='h')]).agg(features).reset_index()
        fin_df['hour'] = df['hour']
        self.locations = fin_df[['locationLatitude', 'locationLongitude']].drop_duplicates()
        self.dataset = fin_df
        # self.length = min(fin_df.groupby(['locationLatitude', 'locationLongitude']).size())
    
    def _get_edge_weights(self):
        num_nodes = len(self.locations)
        self.mapper = {i: tuple(self.locations.iloc[i]) for i in range(num_nodes)}
        self.rev_map = {tuple(self.locations.iloc[i]): i for i in range(num_nodes)}

        edges = []
        weights = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = haversine(tuple(self.locations.iloc[i]), tuple(self.locations.iloc[j]))
                if dist != 0:
                    if self.eq_weights:
                        edges.append([i, j])
                        edges.append([j, i]) 
                        weights.append(1)
                        weights.append(1)
                    else:
                        attr = 1 / (1 + haversine(tuple(self.locations.iloc[i]), tuple(self.locations.iloc[j])))
                        edges.append([i, j])
                        edges.append([j, i]) 
                        weights.append(attr)
                        weights.append(attr)

        self.edges = np.array(edges).T
        self.edge_weights = np.array(weights)

    def _get_task(self, start_pred):
        assert start_pred < 24 and start_pred > 1

        fin_df = self.dataset
        self.features = []
        self.targets = []

        date_list = list(pd.unique(fin_df['time'].dt.date))
        tbar = trange(len(date_list), desc='Creating Dataset')
        # features is per day
        for i in tbar:
            day = date_list[i]
            loc_list = []
            day_df = fin_df[fin_df['time'].dt.date == day]
            
            # numpy array per location per hour
            # array shape: len(locs) x features x 24
            for lat, long in self.locations.itertuples(index=False):
                features_list = []
                loc_df = day_df[(day_df['locationLatitude'] == lat) & (day_df['locationLongitude'] == long)]
                if not loc_df.empty:
                    for hr in range(24):
                        vals = loc_df[loc_df['hour'] == hr]
                        if not vals.empty:
                            features_list.append(np.array(vals.iloc[0, 3:-1]))
                        else:
                            features_list.append(np.zeros(len(fin_df.columns[3:-1])))
                    features_list_np = np.stack(features_list).T
                else:
                    features_list_np = np.zeros((len(fin_df.columns[3:-1]), 24))
                    
                loc_list.append(features_list_np)
            
            fin_arr = np.stack(loc_list).astype('float64')
            self.features.append(fin_arr[:, :, :start_pred])
            self.targets.append(fin_arr[:, :, start_pred:])
    
    def get_dataset(self, start_pred=12, set_dict=None):
        if set_dict:
            self.edges = set_dict['edges']
            self.edge_weights = set_dict['edge_weights']
            self.features = set_dict['features']
            self.targets = set_dict['targets']
        else:
            self._get_edge_weights()
            self._get_task(start_pred)
            
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
    
class AirQualityDataset():
    def __init__(self, path, features=None, lat_long_vals=None, t_range=None, time_zone=None,  interval='hour', eq_weights=True):
        self.eq_weights = eq_weights
        self.interval = interval
        
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df['locationLatitude'] = df['locationLatitude'].round(6)
        df['locationLongitude'] = df['locationLongitude'].round(6)
        cols_to_keep = ['time', 'locationLatitude', 'locationLongitude'] + list(features.keys())

        if features:
            df = df[cols_to_keep]
        if time_zone:
            df.time = df.time.dt.tz_convert(time_zone)
        if t_range:
            df = df[(df['time'] > pd.to_datetime(t_range[0],unit="ns", utc=True)) 
                    & (df['time'] < pd.to_datetime(t_range[1],unit="ns", utc=True))]
        if not lat_long_vals.empty:
            df = df.merge(lat_long_vals, on=['locationLatitude', 'locationLongitude'])

        fin_df = df.groupby(['locationLatitude', 'locationLongitude', pd.Grouper(key='time', freq='h')]).agg(features).reset_index()
        fin_df['hour'] = fin_df['time'].dt.hour
        self.locations = fin_df[['locationLatitude', 'locationLongitude']].drop_duplicates()
        self.dataset = fin_df
        # self.length = min(fin_df.groupby(['locationLatitude', 'locationLongitude']).size())
    
    def _get_edge_weights(self):
        num_nodes = len(self.locations)
        self.mapper = {i: tuple(self.locations.iloc[i]) for i in range(num_nodes)}
        self.rev_map = {tuple(self.locations.iloc[i]): i for i in range(num_nodes)}

        edges = []
        weights = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = haversine(tuple(self.locations.iloc[i]), tuple(self.locations.iloc[j]))
                if dist != 0:
                    if self.eq_weights:
                        edges.append([i, j])
                        edges.append([j, i]) 
                        weights.append(1)
                        weights.append(1)
                    else:
                        attr = 1 / (1 + haversine(tuple(self.locations.iloc[i]), tuple(self.locations.iloc[j])))
                        edges.append([i, j])
                        edges.append([j, i]) 
                        weights.append(attr)
                        weights.append(attr)

        self.edges = np.array(edges).T
        self.edge_weights = np.array(weights)

    def _get_task(self, start_pred):
        assert start_pred < 24 and start_pred > 1

        fin_df = self.dataset
        self.features = []
        self.targets = []

        # features is per day
        for day in list(pd.unique(fin_df['time'].dt.date)):
            loc_list = []
            day_df = fin_df[fin_df['time'].dt.date == day]
            
            # numpy array per location per hour
            # array shape: len(locs) x features x 24
            for lat, long in self.locations.itertuples(index=False):
                features_list = []
                loc_df = day_df[(day_df['locationLatitude'] == lat) & (day_df['locationLongitude'] == long)]
                if not loc_df.empty:
                    for hr in range(24):
                        vals = loc_df[loc_df['hour'] == hr]
                        if not vals.empty:
                            features_list.append(np.array(vals.iloc[0, 3:-1]))
                        else:
                            features_list.append(np.zeros(len(fin_df.columns[3:-1])))
                    features_list_np = np.stack(features_list).T
                else:
                    features_list_np = np.zeros((len(fin_df.columns[3:-1]), 24))
                    
                loc_list.append(features_list_np)
            
            fin_arr = np.stack(loc_list).astype('float64')
            self.features.append(fin_arr[:, :, :start_pred])
            self.targets.append(fin_arr[:, :, start_pred:])
    
    def get_dataset(self, start_pred=12):
        self._get_edge_weights()
        self._get_task(start_pred)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
    
def plot_features(features, y_hat, y_true, feat, save_name):
    # Tensor must be (B, N, F, T)
    preds = np.asarray(y_hat[:, :, feat, :].detach().cpu().numpy())
    labs  = np.asarray(y_true[:, :, feat, :].cpu().numpy())
    nodes = preds.shape[1]

    fig, axs = plt.subplots(nodes//2, 2, figsize=(15,7*np.ceil(nodes/6)))
    print(preds.shape, labs.shape)

    for i, ax in enumerate(axs.flatten()):
        p_mean_val = np.mean(preds[:, i, :], axis=0)
        p_std_val = np.std(preds[:, i, :], axis=0)

        t_mean_val = np.mean(labs[:, i, :], axis=0)
        t_std_val = np.std(labs[:, i, :], axis=0)
        
        ax.plot(p_mean_val, label='Prediction')
        ax.plot(t_mean_val, label='True Label')

        ax.set_xlabel('Hours')
        ax.legend()

    fig.suptitle(f'{list(features.keys())[feat]}')
    fig.tight_layout()
    plt.savefig((f'AirQuality/plots/{list(features.keys())[feat]}_{save_name}.jpeg'))
