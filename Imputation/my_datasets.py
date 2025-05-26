import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import random
from collections import deque
from einops import rearrange
from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.synch_mode import HORIZON
from tsl.datasets.prototypes import DatetimeDataset, TabularDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.ops.framearray import framearray_shape, framearray_to_numpy
from tsl.ops.imputation import to_missing_values_dataset
from tsl.utils import download_url, extract_zip


def add_missing_sensors(dataset: TabularDataset,
                       p_noise=0.05,
                       p_fault=0.01,
                       min_seq=1,
                       max_seq=10,
                       seed=None,
                       inplace=True,
                       masked_sensors = [],
                       connect = None,
                       mode='road'):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    if masked_sensors is None:
        eval_mask = sample_mask(shape,
                            p=p_fault,
                            p_noise=p_noise,
                            mode=mode,
                            adj=dataset.get_connectivity(**connect, layout='dense'))
        
        dataset.p_fault = p_fault
        dataset.p_noise = p_noise
        dataset.min_seq = min_seq
        dataset.max_seq = max_seq
        dataset.seed = seed
        dataset.random = random

        # mask = rearrange(eval_mask, "b n 1 -> b n")
        mask_sum = eval_mask.sum(0)  # n
        masked_sensors = (np.where(mask_sum > 0)[0]).tolist()
    else:
        masked_sensors = list(masked_sensors)
        eval_mask = np.zeros_like(dataset.mask)
        eval_mask[:, masked_sensors] = dataset.mask[:, masked_sensors]

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    test2 = np.sum(dataset.mask, axis=(0))
    test1 = np.sum(eval_mask, axis=(0))

    # Store evaluation mask params in dataset
    return dataset, masked_sensors

class AirQualitySplitter(Splitter):

    def __init__(self,
                 val_len: int = None,
                 test_months: Sequence = (3, 6, 9, 12)):
        super(AirQualitySplitter, self).__init__()
        self._val_len = val_len
        self.test_months = test_months

    def fit(self, dataset):
        nontest_idxs, test_idxs = disjoint_months(dataset,
                                                  months=self.test_months,
                                                  synch_mode=HORIZON)
        # take equal number of samples before each month of testing
        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(nontest_idxs))
        val_len = val_len // len(self.test_months)
        # get indices of first day of each testing month
        delta = np.diff(test_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        end_month_idxs = test_idxs[1:][delta_idxs]
        if len(end_month_idxs) < len(self.test_months):
            end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
        # expand month indices
        month_val_idxs = [
            np.arange(v_idx - val_len, v_idx) - dataset.window
            for v_idx in end_month_idxs
        ]
        val_idxs = np.concatenate(month_val_idxs) % len(dataset)
        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs,
                                                  val_idxs,
                                                  synch_mode=HORIZON,
                                                  as_mask=True)
        train_idxs = nontest_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)

def sample_mask(shape, p=0.002, p_noise=0., mode="random", adj=None):
    assert mode in ["random", "road", "mix", "region"], "The missing mode must be 'random' or 'road' or 'mix'."
    rand = np.random.random
    mask = np.zeros(shape).astype(bool)
    if mode == "random" or mode == "mix":
        mask = mask | (rand(mask.shape) < p)
    if mode == "road" or mode == "mix":
        road_shape = mask.shape[1]
        rand_mask = rand(road_shape) < p_noise
        road_mask = np.zeros(shape).astype(bool)
        road_mask[:, rand_mask] = True
        mask |= road_mask
    if mode == "region":
        num_vert = (rand(mask.shape[1]) < p_noise).sum().item()
        regions = region_masking(adj, num_vert)
        region_mask = np.zeros(shape).astype(bool)
        region_mask[:, regions] = True
        mask |= region_mask

    return mask.astype('uint8')

def region_masking(adj_matrix, n):
    num_nodes = adj_matrix.shape[0]
    if n >= num_nodes:
        return list(range(num_nodes))

    visited = set()
    all_nodes = set(range(num_nodes))

    while len(visited) < n:
        # Start from an unvisited random seed node
        seed = random.choice(list(all_nodes - visited))
        queue = deque([seed])
        visited.add(seed)

        while queue and len(visited) < n:
            current = queue.popleft()
            neighbors = np.nonzero(adj_matrix[current])[0]  # indices with edge

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor.item())
                    queue.append(neighbor)
                    if len(visited) == n:
                        break

    return list(visited)

class AirQualityKrig(DatetimeDataset, MissingValuesMixin):
    r"""Measurements of pollutant :math:`PM2.5` collected by 437 air quality
    monitoring stations spread across 43 Chinese cities from May 2014 to April
    2015.

    The dataset contains also a smaller version :obj:`AirQuality(small=True)`
    with only the subset of nodes containing the 36 sensors in Beijing.

    Data collected inside the `Urban Air
    <https://www.microsoft.com/en-us/research/project/urban-air/>`_ project.

    Dataset size:
        + Time steps: 8760
        + Nodes: 437
        + Channels: 1
        + Sampling rate: 1 hour
        + Missing values: 25.67%

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    url = "https://drive.switch.ch/index.php/s/W0fRqotjHxIndPj/download"

    similarity_options = {'distance'}

    def __init__(self,
                 root: str = None,
                 impute_nans: bool = True,
                 small: bool = False,
                 test_months: Sequence = (3, 6, 9, 12),
                 freq: Optional[str] = None,
                 masked_sensors: Optional[Sequence] = None,
                 p: Optional[float] = 1.):
        self.root = root
        self.small = small
        self.test_months = test_months
        if masked_sensors is None:
            self.masked_sensors = []
        else:
            self.masked_sensors = list(masked_sensors)

        df, mask, eval_mask, dist = self.load(impute_nans=impute_nans, p=p)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='air_quality',
                         name='AQI36' if self.small else 'AQI')
        self.add_covariate('dist', dist, pattern='n n')
        self.set_eval_mask(eval_mask)

    @property
    def raw_file_names(self) -> List[str]:
        return ['full437.h5', 'small36.h5']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['aqi_dist.npy']

    def download(self):
        path = download_url(self.url, self.root_dir, 'data.zip')
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self):
        self.maybe_download()
        # compute distances from latitude and longitude degrees
        path = os.path.join(self.root_dir, 'full437.h5')
        stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(st_coord, to_rad=True).values
        np.save(os.path.join(self.root_dir, 'aqi_dist.npy'), dist)

    def load_raw(self):
        self.maybe_build()
        dist = np.load(os.path.join(self.root_dir, 'aqi_dist.npy'))
        if self.small:
            path = os.path.join(self.root_dir, 'small36.h5')
            eval_mask = pd.read_hdf(path, 'eval_mask')
            dist = dist[:36, :36]
        else:
            path = os.path.join(self.root_dir, 'full437.h5')
            eval_mask = None
        df = pd.read_hdf(path, 'pm25')
        return pd.DataFrame(df), dist, eval_mask

    def load(self, impute_nans=True, p=1.):
        # load readings and stations metadata
        df, dist, eval_mask = self.load_raw()
        # compute the masks:
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is valid
        test = np.sum(eval_mask, axis=(0))

        if len(self.masked_sensors):
            eval_mask = np.zeros(mask.shape)
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
        else:
            eval_mask = sample_mask(mask.shape,
                                    p=0.,
                                    p_noise=p,
                                    mode="road")
        # eventually replace nans with weekly mean by hour
        
        if impute_nans:
            from tsl.ops.framearray import temporal_mean
            df = df.fillna(temporal_mean(df))
        
        test2 = np.sum(mask, axis=(0))
        test1 = np.sum(eval_mask, axis=(0))
        return df, mask, eval_mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'air_quality':
            val_len = kwargs.get('val_len')
            return AirQualitySplitter(test_months=self.test_months,
                                      val_len=val_len)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            from tsl.ops.similarities import gaussian_kernel

            # use same theta for both air and air36
            theta = np.std(self.dist[:36, :36])
            return gaussian_kernel(self.dist, theta=theta)

class AirQualitySmaller(DatetimeDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self,
                 root: str = None,
                 impute_nans: bool = True,
                 test_months: Sequence = (3, 6, 9),
                 infer_eval_from: str = 'next',
                 features: list = ['PM2.5'],
                 freq: Optional[str] = None,
                 masked_sensors: Optional[Sequence] = None):
        self.root = root
        self.test_months = test_months
        self.infer_eval_from = infer_eval_from  # [next, previous]
        self.features = features
        if masked_sensors is None:
            self.masked_sensors = []
        else:
            self.masked_sensors = list(masked_sensors)
        df, mask, eval_mask, dist = self.load(impute_nans=impute_nans)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='air_quality',
                         name='AQI12')
        
        self.add_covariate('dist', dist, pattern='n n')

        eval_mask = self._parse_target(eval_mask)
        eval_mask = framearray_to_numpy(eval_mask).astype(bool)
        self.add_covariate('eval_mask', eval_mask, 't n f')

        # self.df = df
        # self.masks = mask
        # self.eval_masks = eval_mask
        # self.distance = dist

    @property
    def raw_file_names(self) -> List[str]:
        return ['merged_full.csv']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['aqi_dist.npy']

    def build(self):
        # compute distances from latitude and longitude degrees
        path = os.path.join(self.root_dir, 'merged_full.csv')
        stations = pd.DataFrame(pd.read_csv(path))
        stations = stations.drop_duplicates(subset=["station"])[["station", "locationLatitude", "locationLongitude"]]
        st_coord = stations.loc[:, ['locationLatitude', 'locationLongitude']]
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(st_coord, to_rad=True).values
        np.save(os.path.join(self.root_dir, 'aqi_dist.npy'), dist)

    def load_raw(self):
        self.maybe_build()
        dist = np.load(os.path.join(self.root_dir, 'aqi_dist.npy'))
        path = os.path.join(self.root_dir, 'merged_full.csv')
        eval_mask = None
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df_pivot = df.pivot(index="datetime", columns="station", values=self.features)
        df_pivot.columns.names = ["channels", "nodes"]
        df_pivot.columns = df_pivot.columns.swaplevel(0, 1) 
        df_pivot.sort_index(axis=1, level=0, inplace=True)
        df_pivot = df_pivot.rename(columns={feat: ind for ind, feat in enumerate(self.features)})
        return pd.DataFrame(df_pivot), dist, eval_mask

    def load(self, impute_nans=True):
        # load readings and stations metadata
        df, dist, eval_mask = self.load_raw()
        # compute the masks:
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is valid
        if eval_mask is None:
            eval_mask = np.zeros((mask.shape))
        # 1 if value is ground-truth for imputation
        if len(self.masked_sensors):
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
            mask[:, self.masked_sensors] = 0
        # eventually replace nans with weekly mean by hour
        # print(np.sum(eval_mask, axis=0))
        if impute_nans:
            from tsl.ops.framearray import temporal_mean
            df = df.fillna(temporal_mean(df))
        return df, mask, eval_mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'air_quality':
            val_len = kwargs.get('val_len')
            return AirQualitySplitter(test_months=self.test_months,
                                      val_len=val_len)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            from tsl.ops.similarities import gaussian_kernel

            # use same theta for both air and air36
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)
        
## Auckland Dataset

import itertools

COLS = ['pm10ConcNumIndividual.value', 'pm1ConcNumIndividual.value',
        'pm2_5ConcNumIndividual.value', 'relHumidInternalIndividual.value']
AUCKLAND = {
    'df' :      pd.DataFrame({
                'locationLatitude': [-36.844079, -36.844113, -36.711932, -36.898491, -36.906652, -36.876728],
                'locationLongitude': [174.762123, 174.761371, 174.740808, 174.591428, 174.633079, 174.703081]}), 
    'timezone': 'Pacific/Auckland'}

INVERCARGILL2 = {
    'df' :      pd.DataFrame({
                'locationLongitude': [168.354731, 168.350339, 168.350151, 168.374574, 168.387039, 168.350258, 168.381864,
                                     168.375167, 168.350805, 168.377209, 168.382873, 168.384734, 168.361357, 168.375977,
                                     168.35045, 168.349358, 168.346235, 168.361723, 168.386655, 168.366703, 168.361048, 
                                     168.374085, 168.350047, 168.370799, 168.353385, 168.366792, 168.361174, 168.383326,
                                     168.369778, 168.360898, 168.360781, 168.38856, 168.360558, 168.369855, 168.36128,
                                     168.355503, 168.379932, 168.375381, 168.366307, 168.377629, 168.354625, 168.374201],

                'locationLatitude': [-46.423463, -46.391143, -46.404305, -46.403735, -46.435166, -46.391083, -46.402722,
                                     -46.396632, -46.395459, -46.423565, -46.385037, -46.391359, -46.38417, -46.416871,
                                     -46.384261, -46.378189, -46.379938, -46.396574, -46.423486, -46.423553, -46.410818,
                                     -46.403778, -46.404272, -46.410898, -46.41079, -46.430023, -46.390393, -46.397899,
                                     -46.430981, -46.442108, -46.43678, -46.417054, -46.375673, -46.431264, -46.404628, 
                                     -46.416806, -46.409627, -46.396528, -46.417347, -46.430643, -46.429926, -46.390498]}), 
    'timezone': 'Pacific/Auckland'}

INVERCARGILL1 = {
    'df' :      pd.DataFrame({
                'locationLongitude': [168.382115, 168.354731, 168.367298, 168.387039, 168.372177, 168.382602, 168.354712,
                                      168.359962, 168.377209, 168.359915, 168.375977, 168.38748, 168.386655, 168.366703,
                                      168.360128, 168.377406, 168.382387, 168.354391, 168.376304, 168.371295, 168.372183,
                                      168.366792, 168.35456, 168.371516, 168.366803, 168.371293, 168.387123, 168.382709,
                                      168.38856, 168.387645, 168.377232, 168.360316, 168.355503, 168.381202, 168.359866,
                                      168.359854, 168.377629, 168.354625, 168.366307, 168.382259, 168.371009],

                'locationLatitude': [-46.42718, -46.423463, -46.433992, -46.435166, -46.430401, -46.420204, -46.420094,
                                     -46.426834, -46.423565, -46.420217, -46.416871, -46.430598, -46.423486, -46.423553,
                                     -46.430016, -46.419827, -46.429942, -46.426854, -46.434382, -46.420081, -46.427286,
                                     -46.430023, -46.434065, -46.43433, -46.427105, -46.42341, -46.420234, -46.416867,
                                     -46.417054, -46.426916, -46.426421, -46.434034, -46.416806, -46.43481, -46.416669,
                                     -46.423492, -46.430643, -46.429926, -46.417347, -46.423572, -46.417033]}), 
    'timezone': 'Pacific/Auckland'}

LOCATIONS = ['Auckland', 'Invercargill1', 'Invercargill2']

def AirQualityCreate(path, agg_func = 'mean', features=None, t_range=None, location='Auckland'):
    for feat in features:
        assert feat in COLS

    assert agg_func in ['mean', 'max', 'min']
    features = {feat:agg_func for feat in features}

    assert location in LOCATIONS, f'Locations must be {LOCATIONS}'
    if location == 'Auckland':
        lat_long_vals = AUCKLAND["df"]
    elif location == 'Invercargill1':
        lat_long_vals = INVERCARGILL1['df']
    elif location == 'Invercargill2':
        lat_long_vals = INVERCARGILL2['df']

    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['time'], utc=True)
    df['locationLatitude'] = df['locationLatitude'].round(6)
    df['locationLongitude'] = df['locationLongitude'].round(6)
    cols_to_keep = ['datetime', 'locationLatitude', 'locationLongitude'] + list(features.keys())

    # Clean dataset
    if features:
        df = df[cols_to_keep]
    if t_range:
        df = df[(df['datetime'] > pd.to_datetime(t_range[0],unit="ns", utc=True)) 
                & (df['datetime'] < pd.to_datetime(t_range[1],unit="ns", utc=True))]
    if not lat_long_vals.empty:
        df = df.merge(lat_long_vals, on=['locationLatitude', 'locationLongitude'])

    fin_df = df.groupby([pd.Grouper(key='datetime', freq='h'), 'locationLatitude', 'locationLongitude']).agg(features).reset_index()

    unique_stations = fin_df[['locationLatitude', 'locationLongitude']].drop_duplicates().dropna().reset_index(drop=True)
    unique_stations['station'] = range(1, len(unique_stations) + 1)  
    
    fin_df = fin_df.merge(unique_stations, on=['locationLatitude', 'locationLongitude'], how='left')

    # Shape daset
    unique_datetimes = fin_df["datetime"].unique()

    datetime_range = pd.date_range(start=np.min(unique_datetimes), end=np.max(unique_datetimes), freq='h')
    unique_stations = fin_df["station"].unique()

    all_combinations = pd.DataFrame(
        list(itertools.product(datetime_range, unique_stations)),
        columns=["datetime", "station"]
    )

    df_complete = all_combinations.merge(fin_df, on=["datetime", "station"], how="left")
    df_complete[['locationLatitude', 'locationLongitude']] = \
        df_complete.groupby('station')[['locationLatitude', 'locationLongitude']].transform(lambda x: x.ffill().bfill())

    return df_complete

# niwa_df = AirQualityCreate('../../../AirData/Niwa/allNIWA_clarity.csv', ['pm2_5ConcNumIndividual.value', 'relHumidInternalIndividual.value'], ['2022-04-01', '2022-12-01'])

class AirQualityAuckland(DatetimeDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self,
                 root: str = None,
                 impute_nans: bool = True,
                 test_months: Sequence = (7, 8),
                 infer_eval_from: str = 'next',
                 features: list = ['pm2_5ConcNumIndividual.value'],
                 agg_func: str = 'mean',
                 location: str = 'Auckland',
                 t_range: Optional[list] = None,
                 freq: Optional[str] = None,
                 masked_sensors: Optional[Sequence] = None,
                 p: Optional[float] = 1.):
        self.root = root
        self.test_months = test_months
        self.infer_eval_from = infer_eval_from  # [next, previous]
        self.features = features
        self.t_range = t_range
        self.agg_func = agg_func
        self.location = location

        if masked_sensors is None:
            self.masked_sensors = []
        else:
            self.masked_sensors = list(masked_sensors)

        if location == 'Auckland':
            self.save_p = 'auck_aqi_dist'
        elif location == 'Invercargill1':
            self.save_p = 'invg1_aqi_dist'
        elif location == 'Invercargill2':
            self.save_p = 'invg2_aqi_dist'
        
        df, mask, eval_mask, dist = self.load(impute_nans=impute_nans, p=p)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='air_quality',
                         name='AQI12')
        
        self.add_covariate('dist', dist, pattern='n n')
        self.set_eval_mask(eval_mask)

        self.df = df
        # self.masks = mask
        # self.eval_masks = eval_mask
        # self.distance = dist

    @property
    def raw_file_names(self) -> List[str]:
        return ['allNIWA_clarity.csv']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + [f'{self.save_p}.npy']

    def build(self):
        # compute distances from latitude and longitude degrees
        path = os.path.join(self.root_dir, 'allNIWA_clarity.csv')
        stations = AirQualityCreate(path, self.agg_func, self.features, self.t_range, self.location)
        stations = stations.drop_duplicates(subset=["station"])[["station", "locationLatitude", "locationLongitude"]]
        self.stations = stations

        st_coord = stations.loc[:, ['locationLatitude', 'locationLongitude']]
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(st_coord, to_rad=True).values
        np.save(os.path.join(self.root_dir, f'{self.save_p}.npy'), dist)

    def load_raw(self):
        self.maybe_build()
        dist = np.load(os.path.join(self.root_dir, f'{self.save_p}.npy'))
        path = os.path.join(self.root_dir, 'allNIWA_clarity.csv')
        eval_mask = None
        df = AirQualityCreate(path, self.agg_func, self.features, self.t_range, self.location)
        stations = df.drop_duplicates(subset=["station"])[["station", "locationLatitude", "locationLongitude"]]
        self.stations = stations

        df_pivot = df.pivot(index="datetime", columns="station", values=self.features)
        df_pivot.columns.names = ["channels", "nodes"]
        df_pivot.columns = df_pivot.columns.swaplevel(0, 1) 
        df_pivot.sort_index(axis=1, level=0, inplace=True)
        df_pivot = df_pivot.rename(columns={feat: ind for ind, feat in enumerate(self.features)})
        
        return pd.DataFrame(df_pivot), dist, eval_mask

    def load(self, impute_nans=True, p=1.):
        # load readings and stations metadata
        df, dist, eval_mask = self.load_raw()
        # compute the masks:
        mask = ((~np.isnan(df.values)) & (df.values != 0)).astype('uint8')  # 1 if value is valid
        if eval_mask is None:
            eval_mask = np.zeros((mask.shape))
        # 1 if value is ground-truth for imputation
        if len(self.masked_sensors):
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
        else:
            eval_mask = sample_mask(mask.shape,
                                    p=0.,
                                    p_noise=p,
                                    mode="road")
            
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            from tsl.ops.framearray import temporal_mean
            df = df.fillna(temporal_mean(df))
        return df, mask, eval_mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'air_quality':
            val_len = kwargs.get('val_len')
            return AirQualitySplitter(test_months=self.test_months,
                                      val_len=val_len)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            from tsl.ops.similarities import gaussian_kernel

            # use same theta for both air and air36
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)