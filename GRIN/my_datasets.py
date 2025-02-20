import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.synch_mode import HORIZON
from tsl.datasets.prototypes import DatetimeDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.utils import download_url, extract_zip
from tsl.ops.framearray import framearray_shape, framearray_to_numpy


def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1
    if it is present in the DataFrame and absent in the :obj:`infer_from` month.

    Args:
        df (pd.Dataframe): The DataFrame.
        infer_from (str): Denotes from which month the evaluation value must be
            inferred. Can be either :obj:`previous` or :obj:`next`.

    Returns:
        pd.DataFrame: The evaluation mask for the DataFrame.
    """
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns,
                             data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('`infer_from` can only be one of {}'.format(
            ['previous', 'next']))
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
        mask_j = mask[cond_j]
        offset_i = 12 * (year_i - year_j) + (month_i - month_j)
        mask_i = mask_j.shift(1, pd.DateOffset(months=offset_i))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        i_idx = mask_i.index
        eval_mask.loc[i_idx] = ~mask_i.loc[i_idx] & mask.loc[i_idx]
    return eval_mask


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


class AirQualitySmaller(DatetimeDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self,
                 root: str = None,
                 impute_nans: bool = True,
                 test_months: Sequence = (3, 6, 9, 12),
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

def AirQualityCreate(path, features=None, t_range=None):
    for feat in features:
        assert feat in COLS
    
    features = {feat:'mean' for feat in features}

    lat_long_vals = AUCKLAND["df"]

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
                 t_range: Optional[list] = None,
                 freq: Optional[str] = None,
                 masked_sensors: Optional[Sequence] = None):
        self.root = root
        self.test_months = test_months
        self.infer_eval_from = infer_eval_from  # [next, previous]
        self.features = features
        self.t_range = t_range

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

        self.df = df
        # self.masks = mask
        # self.eval_masks = eval_mask
        # self.distance = dist

    @property
    def raw_file_names(self) -> List[str]:
        return ['allNIWA_clarity.csv']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['auck_aqi_dist.npy']

    def build(self):
        # compute distances from latitude and longitude degrees
        path = os.path.join(self.root_dir, 'allNIWA_clarity.csv')
        stations = AirQualityCreate(path, self.features, self.t_range)
        stations = stations.drop_duplicates(subset=["station"])[["station", "locationLatitude", "locationLongitude"]]
        st_coord = stations.loc[:, ['locationLatitude', 'locationLongitude']]
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(st_coord, to_rad=True).values
        np.save(os.path.join(self.root_dir, 'auck_aqi_dist.npy'), dist)

    def load_raw(self):
        self.maybe_build()
        dist = np.load(os.path.join(self.root_dir, 'auck_aqi_dist.npy'))
        path = os.path.join(self.root_dir, 'allNIWA_clarity.csv')
        eval_mask = None
        df = AirQualityCreate(path, self.features, self.t_range)

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