import os.path as osp

from torch_geometric.data.temporal import TemporalData

import torch
import networkx as nx
import pandas as pd
import numpy as np

#TODO: Create this dataset 

class StarDataset(TemporalData):
    def __init__(self, src: torch.Tensor | None = None, dst: torch.Tensor | None = None, t: torch.Tensor | None = None, msg: torch.Tensor | None = None, **kwargs):
        super().__init__(src, dst, t, msg, **kwargs)

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
        features = np.zeros((len(data_events), 4 + 2))
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


            # Even types
            event = data['event_type']
            y[ind][event_dict[event]] = 1

        _, indexes = np.unique(np.concatenate((src, dst)), return_inverse=True)
        src, dst = np.split(indexes, 2)

        src = torch.Tensor(src).type(torch.long)
        dst = torch.Tensor(dst).type(torch.long)
        t = torch.Tensor(t).type(torch.long)
        y = torch.Tensor(y)

        data = TemporalData(src=src, dst=dst, t=t, y=y)