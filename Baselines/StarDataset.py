import os.path as osp

from torch_geometric.data.temporal import TemporalData

import torch
import networkx as nx
import pandas as pd
import numpy as np
import pyarrow
import tqdm

class StarDataset():
    def __init__(self, event_path, data_path, date_range, with_fish, self_loop, device):
        data_events = pd.read_parquet(event_path)
        data_vessels = pd.read_csv(data_path)

        if with_fish:
            events = ['FISH', 'PORT', 'ECTR']
        else:
            events = ['PORT', 'ECTR']
            data_events = data_events[data_events.event_type != 'FISH']
            
        event_dict = dict(zip(events, range(len(events))))
        self.rev_event_dict = dict(zip(range(len(events)), events))
        feature_dict = {x['vessel_id']: x['label'] for _, x in data_vessels.iterrows()}

        # convert timesteps
        data_events['start_time'] = pd.to_datetime(data_events['start_time'], format='%d/%m/%Y %H:%M')
        data_events = data_events.sort_values(by='start_time').reset_index(drop=True)

        # Filter dates
        if date_range == 'all':
            pass
        else:
            mask = (data_events['start_time'] >= date_range[0]) & (data_events['start_time'] <= date_range[1])
            data_events = data_events.loc[mask].reset_index(drop=True)
        
        timesteps = (data_events['start_time'] - data_events['start_time'][0]).dt.days

        # Initialise inputs 
        src = np.zeros((len(data_events)))
        dst = np.zeros((len(data_events)))
        t = np.zeros((len(data_events)))
        # features = (lat_n,lat_s,lon_e,lon_w,ves1,ves2)
        features = np.zeros((len(data_events), 4))
        y = np.zeros((len(data_events), len(events)))

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
            else:
                if with_fish:
                    if self_loop:
                        dst[ind] = data['vessel_id']
                    else:
                        dst[ind] = 0

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

        src = torch.Tensor(src).type(torch.long).to(device)
        dst = torch.Tensor(dst).type(torch.long).to(device)
        t = torch.Tensor(t).type(torch.long).to(device)
        features = torch.Tensor(features).to(device)
        y = torch.Tensor(y).to(device)

        self.data = TemporalData(src=src, dst=dst, t=t, msg=features, y=y)
        self.ind_map = {vals[i]: i for i in range(len(vals))}
        self.rev_ind_map = {i: vals[i] for i in range(len(vals))}

    def get_data(self):
        return self.data
    
    def get_ind_map(self):
        return self.ind_map
    
    def get_rev_ind_map(self):
        return self.rev_ind_map
    
    def get_rev_event_dict(self):
        return self.rev_event_dict
        