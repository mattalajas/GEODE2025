import math
from typing import Iterable, Tuple
from functools import partial

import networkx as nx
import numpy as np
import torch
import tsl
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data.storage import recursive_apply
from tsl.data import ImputationDataset
from tsl.data.preprocessing import Scaler, ScalerModule
from tsl.datasets.prototypes import TabularDataset
from tsl.metrics import numpy as numpy_metrics
from tsl.ops.imputation import to_missing_values_dataset
from tsl.ops.pattern import broadcast, outer_pattern, take
from tsl.typing import TensArray


def zeros_to_one_(scale):
    """Set to 1 scales of near constant features, detected by identifying
    scales close to machine precision, in place.
    Adapted from :class:`sklearn.preprocessing._data._handle_zeros_in_scale`
    """
    if np.isscalar(scale):
        return 1.0 if np.isclose(scale, 0.) else scale
    eps = 10 * np.finfo(scale.dtype).eps
    zeros = np.isclose(scale, 0., atol=eps, rtol=eps)
    scale[zeros] = 1.0
    return scale


def fit_wrapper(fit_function):

    def fit(obj: "Scaler", x, *args, **kwargs) -> "Scaler":
        x_type = type(x)
        x = np.asarray(x)
        fit_function(obj, x, *args, **kwargs)
        if x_type is Tensor:
            obj.torch()
        return obj

    return fit

def closest_distances_unweighted(G, source_nodes, target_nodes):
    result = {}
    target_set = set(target_nodes)
    
    for source in source_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        distances = [lengths[t] for t in target_set if t in lengths]
        result[source] = min(distances) if distances else float('inf')
    
    return result

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    sum_of_diff_square = ((x1-x2)**2).sum(-1) + 1e-8
    return sum_of_diff_square.sqrt()

def moment_diff(sx1, sx2, k, og_batch, coarse_batch):
    """
    difference between moments
    """
    ss1 = scatter(sx1**k, og_batch, dim=0, dim_size=None, reduce='mean')
    ss2 = scatter(sx2**k, coarse_batch, dim=0, dim_size=None, reduce='mean')
    return l2diff(ss1,ss2)

def cmd(x1, x2, og_batch, coarse_batch, n_moments=2):
    """
    central moment discrepancy (cmd)
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    #print("input shapes", x1.shape, x2.shape)
    mx1 = scatter(x1, og_batch, dim=0, dim_size=None, reduce='mean')
    mx2 = scatter(x2, coarse_batch, dim=0, dim_size=None, reduce='mean')
    #print("mx* shapes should be same (batch_szie, dim)", mx1.shape, mx2.shape)
    sx1 = x1 - mx1.repeat_interleave(torch.unique(og_batch, return_counts=True)[1], dim=0)
    sx2 = x2 - mx2.repeat_interleave(torch.unique(coarse_batch, return_counts=True)[1], dim=0)
    #print("sx1, sx2 should be same size as input", sx1.shape, sx2.shape)
    dm = l2diff(mx1, mx2)
    #print("dm should have shape (batch_size,)", dm.shape)
    scms = dm
    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms = scms + moment_diff(sx1, sx2, i+2, og_batch, coarse_batch)
    return scms

def test_wise_eval(y_hat, y_true, mask, known_nodes, adj, mode, num_groups=4, alpha = 0.20):
    numpy_graph = nx.from_numpy_array(adj)
    k_nodes = np.array(known_nodes)
    u_nodes = np.array([i for i in range(adj.shape[0]) if i not in known_nodes])
    m_adj = (adj > 0).astype(float)
    group_size = u_nodes.shape[0] // num_groups

    # LPS
    n = adj.shape[-1]

    A_hat = m_adj
    idx = np.arange(n)
    A_hat[idx, idx] = 1
    D = np.diag(np.sum(A_hat, axis=1))

    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    I = np.eye(n)

    P = np.linalg.inv((I - (1-alpha)*A_norm))
    T = np.zeros((n, n))
    T[k_nodes, k_nodes] = 1
    ones = np.ones((n,))

    LPS = P @ T @ ones

    sorted_lps = sorted(u_nodes, key=lambda i: LPS[i])
    lps_gr = [sorted_lps[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_lps) % num_groups
    if remainder:
        lps_gr[-1].extend(sorted_lps[-remainder:])

    # CC
    closeness = nx.closeness_centrality(numpy_graph)
    closeness = {node: score for node, score in closeness.items() if score > 0}

    sorted_cls = sorted([i for i in u_nodes if i in closeness], key=lambda i: closeness[i])
    # sorted_cls = [x for x, _ in sorted_cls]
    cls_gr = [sorted_cls[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_cls) % num_groups
    if remainder:
        cls_gr[-1].extend(sorted_cls[-remainder:])

    # KHR
    khr_grouped = closest_distances_unweighted(numpy_graph, u_nodes, k_nodes.tolist())
    khr_grouped = {node: score for node, score in khr_grouped.items() if score < 1e9}
    khr_gr = [[] for _ in range(num_groups)]

    for key, pos in khr_grouped.items():
        value = pos-1
        if value < num_groups:
            khr_gr[value].append(key)
        else:
            khr_gr[num_groups-1].append(key)

    # Evaluate
    group_dict = {'LPS': lps_gr,
                'CC': cls_gr,
                'KHR': khr_gr}
    res = {f'{mode}_mae': numpy_metrics.mae(y_hat, y_true, mask),
               f'{mode}_mre': numpy_metrics.mre(y_hat, y_true, mask),
               f'{mode}_rmse': numpy_metrics.rmse(y_hat, y_true, mask)}

    for key, groups in group_dict.items():
        results = {'mae':[], 'mre':[], 'rmse':[]}
        for group in groups:
            node_mask = np.zeros_like(mask, dtype=bool)
            if len(node_mask.shape) == 4:
                node_mask[:, :, group] = True
            elif len(node_mask.shape) == 3:
                node_mask[:, group] = True
            else:
                raise 'node_mask dim only 3 or 2'
            
            masked_adj = mask * node_mask

            results['mae'].append(numpy_metrics.mae(y_hat, y_true, masked_adj))
            results['mre'].append(numpy_metrics.mre(y_hat, y_true, masked_adj))
            results['rmse'].append(numpy_metrics.rmse(y_hat, y_true, masked_adj))

        for metric, val in results.items():
            res[f'max_{metric}_{key}_{mode}'] = max(val)
            res[f'min_{metric}_{key}_{mode}'] = min(val)
    
    return res    

def add_missing_sensors(dataset: TabularDataset,
                       p_noise=0.05,
                       p_fault=0.01,
                       min_seq=1,
                       max_seq=10,
                       seed=None,
                       inplace=True,
                       masked_sensors = [],
                       connect = None,
                       spatial_shift = False, 
                       order = 0,
                       node_features = 'CC',
                       mode='road'):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    if masked_sensors is None:
        if spatial_shift:
            eval_mask = shift_mask(shape, feature=node_features, order=order, 
                                   adj=dataset.get_connectivity(**connect, layout='dense'),
                                   p_noise=p_noise)
            dataset.seed = seed
        else:
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

def shift_mask(shape, feature, order, adj, p_noise=0.05):
    mask = np.zeros(shape).astype(bool)
    
    try:
        adj = adj.numpy()
    except:
        pass

    G = nx.from_numpy_array(adj)
    parts = math.ceil(adj.shape[0]*p_noise)

    if feature == 'CC':
        # Compute closeness centrality
        closeness = nx.closeness_centrality(G)
        nonzero_c = {node: score for node, score in closeness.items() if score > 0}

        # Sort nodes by closeness centrality in descending order
        sorted_nodes = sorted(nonzero_c.items(), key=lambda x: x[1])
        ord_nodes = [x for x, _ in sorted_nodes]

        f_nodes = ord_nodes[parts*order:parts*(order+1)]
        f_nodes_mask = np.zeros(shape).astype(bool)
        f_nodes_mask[:, f_nodes] = True
        mask |= f_nodes_mask
        
    elif feature == 'ND':
        degree = dict(nx.degree(G))
        nonzero_d = {node: score for node, score in degree.items() if score > 0}

        # Sort nodes by node degree in descending order
        sorted_nodes = sorted(nonzero_d.items(), key=lambda x: x[1])
        ord_nodes = [x for x, _ in sorted_nodes]

        f_nodes = ord_nodes[parts*order:parts*(order+1)]
        f_nodes_mask = np.zeros(shape).astype(bool)
        f_nodes_mask[:, f_nodes] = True
        mask |= f_nodes_mask
    else:
        raise f"{feature} not implemented"
    
    return mask.astype('uint8')

def sample_mask(shape, p=0.002, p_noise=0., mode="random", adj=None):
    assert mode in ["random", "road", "mix"], "The missing mode must be 'random' or 'road' or 'mix'."
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
    return mask.astype('uint8')

# Code retrieved from TSL library <https://github.com/TorchSpatiotemporal/tsl>

import os
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tsl.datasets.prototypes import DatetimeDataset
from tsl.datasets.prototypes.casting import to_pandas_freq
from tsl.utils import download_url, extract_zip

__base_url__ = "https://drive.switch.ch/index.php/s/nJgK7ca28hk7AMU/download"
__subsets__ = ["CA", "GBA", "GLA", "SD"]
SubsetType = Literal["CA", "GBA", "GLA", "SD"]


class LargeST(DatetimeDataset):
    r"""LargeST is a large-scale traffic forecasting dataset containing 5 years
    of traffic readings from 01/01/2017 to 12/31/2021 collected every 5 minutes
    by 8600 traffic sensors in California.

    Given the large number of sensors in the dataset, there are 3 subsets of
    sensors that can be selected:

    + :obj:`GLA` (Greater Los Angeles)
        + Nodes: 3834
        + Edges: 98703
        + District: 7, 8, 12

    + :obj:`GBA` (Greater Bay Area)
        + Nodes: 2352
        + Edges: 61246
        + District: 4

    + :obj:`SD` (San Diego)
        + Nodes: 716
        + Edges: 17319
        + District: 11

    By default, the full dataset :obj:`CA` is loaded, corresponding to the
    whole California.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). Introduced in the paper
    `"LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting"
    <https://arxiv.org/abs/2306.08259>`_ (Liu et al., 2023),
    where only readings from 2019 are considered, aggregated into 15-minutes
    intervals.

    Dataset information:
        + Time steps: 525888
        + Nodes: 8600
        + Edges: 201363
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 1.51%

    Static attributes:
        + :obj:`metadata`: storing for each node:
            + ``lat``: latitude of the sensor;
            + ``lon``: longitude of the sensor;
            + ``district``: California's district where sensor is located (one
              of ``3``, ``4``, ``5``, ``6``, ``7``, ``8``, ``10``, ``11``,
              ``12``);
            + ``county``: California's county where sensor is located;
            + ``fwy_id``: id of highway where a sensor is located;
            + ``n_lanes``: the number of lanes in correspondence to the sensor
              (max 8);
            + ``direction``: direction of the highway measured by the sensor
              (one of ``N``, ``S``, ``E``, ``W``).
        + :obj:`adj`: weighted adjacency matrix
          :math:`\mathbf{A} \in \mathbb{R}^{N \times N}` built using road
          distances.

    Args:
        root (str, optional): The root directory where data will be downloaded
            and stored. If :obj:`None`, then defaults to :obj:`.storage` folder
            inside :tsl:`null` tsl's root directory.
            (default: :obj:`None`)
        subset (str): The subset to be loaded. Must be one of :obj:`"CA"`,
            :obj:`"GLA"`, :obj:`"GBA"`, :obj:`"SD"`.
            (default: :obj:`"CA"`)
        year (int or list): The year(s) to be loaded. Must be (a list) in
            :obj:`[2017, 2021]`. Note that raw data are divided by year and
            only requested years are downloaded.
            (default: :obj:`2019`)
        imputation_mode (str, optional): How to impute missing values. If
            :obj:`"nearest"`, then use nearest observation; if :obj:`"zero"`,
            fill missing values with :obj:`0`; if :obj:`None`, do not impute
            (leave :obj:`nan`).
            (default: :obj:`"zero"`)
        freq (str): The sampling rate used for resampling (e.g., :obj:`"15T"`
            for 15-minutes intervals resampling).
            (default: :obj:`"15T"`)
        precision (int or str): The float precision of the dataset.
            (default: :obj:`32`)
    """
    base_url = __base_url__
    url = {
        "2017": __base_url__ + "?path=%2F2017&files=data.h5",
        "2018": __base_url__ + "?path=%2F2018&files=data.h5",
        "2019": __base_url__ + "?path=%2F2019&files=data.h5",
        "2020": __base_url__ + "?path=%2F2020&files=data.h5",
        "2021": __base_url__ + "?path=%2F2021&files=data.h5",
        "sensors": __base_url__ + "?files=sensors.zip",
    }

    similarity_options = {"precomputed"}

    def __init__(self,
                 root: str = None,
                 subset: SubsetType = "CA",
                 year: Optional[Union[int, Sequence[int]]] = 2019,
                 imputation_mode: Literal["nearest", "zero", None] = "zero",
                 freq: str = "15T",
                 precision: Union[int, str] = 32):
        # set root path
        self.root = root

        subset = subset.upper()
        if subset not in __subsets__:
            raise ValueError(
                f"Incorrect choice for 'subset' ({subset}). "
                f"Available options are {', '.join(__subsets__)}.")
        self.subset = subset

        view_years = years_set = set(range(2017,
                                           2022))  # between 2017 and 2021
        if year is not None:
            year = {year} if isinstance(year, int) else set(year)
            view_years = view_years.intersection(year)
            if not len(view_years):
                raise ValueError(f"Incorrect choice for 'year' ({year}). "
                                 f"Must be a subset of {years_set}.")
        self.years = sorted(view_years)

        self.imputation_mode = imputation_mode
        assert imputation_mode in ["nearest", "zero", None]

        # Set dataset frequency here to resample when loading
        if freq is not None:
            freq = to_pandas_freq(freq)
        self.freq = freq

        # load dataset
        readings, mask, metadata, adj = self.load()
        covariates = {"metadata": (metadata, 'n f'), "adj": (adj, 'n n')}
        super().__init__(target=readings,
                         freq=freq,
                         mask=mask,
                         covariates=covariates,
                         similarity_score="precomputed",
                         temporal_aggregation="mean",
                         spatial_aggregation="mean",
                         name=f"LargeST-{subset}",
                         precision=precision)

    @property
    def raw_file_names(self) -> Dict[str, str]:
        out = {
            str(year): os.path.join(str(year), "data.h5")
            for year in self.years
        }
        out["metadata"] = os.path.join("sensors", "metadata.csv")
        out["adj"] = os.path.join("sensors", "adj.npz")
        return out

    def download(self) -> None:
        for key, filepath in self.raw_files_paths.items():
            # download only required data that are missing
            if not os.path.exists(filepath):
                # "metadata" and "adj" are inside single .zip file
                if key in ["metadata", "adj"]:
                    sub_dir = os.path.dirname(filepath)
                    os.makedirs(sub_dir, exist_ok=True)
                    # download, extract, and remove .zip file
                    in_dir = download_url(self.url["sensors"],
                                          sub_dir,
                                          filename="sensors.zip")
                    extract_zip(in_dir, sub_dir)
                    os.unlink(in_dir)
                else:  # download directly .h5 file containing readings per year
                    sub_dir, filename = os.path.split(filepath)
                    os.makedirs(sub_dir, exist_ok=True)
                    download_url(self.url[key], sub_dir, filename)

    def load_raw(self):
        self.maybe_download()

        filenames = self.required_files_paths

        # load sensors information
        metadata = pd.read_csv(filenames["metadata"], index_col=0)
        max_nodes = len(metadata)

        # possibly select subset, "CA" stands for no subset (whole California)
        node_mask = slice(None)
        if self.subset == "GLA":  # Greater Los Angeles
            node_mask = ((metadata.district == 7) | (metadata.district == 8) |
                         (metadata.district == 12)).values
        elif self.subset == "GBA":  # Greater Bay Area
            node_mask = (metadata.district == 4).values
        elif self.subset == "SD":  # San Diego
            node_mask = (metadata.district == 11).values
        metadata = metadata.loc[node_mask]

        # load traffic data only for requested years
        readings = []
        for year in self.years:
            data_path = filenames[str(year)]
            data_df = pd.read_hdf(data_path, key="readings")
            data_df = data_df.loc[:, node_mask]  # filter subset
            # resample here to aggregate only valid observations and
            # align to authors' preprocessing
            if self.freq is not None:
                data_df = data_df.resample(self.freq).mean()
                # in authors' code: data_df.resample('15T').mean().round(0)
            readings.append(data_df)

        readings = (
            readings[0] if len(readings) == 1  # avoid useless
            else pd.concat(readings, axis=0))  # computations

        # load adjacency
        edge_index, edge_weight = np.load(filenames["adj"]).values()
        # build square adj from coo to add adj as covariate
        adj = np.eye(max_nodes, dtype=np.float32)
        adj[tuple(edge_index)] = edge_weight
        adj = adj[node_mask][:, node_mask]

        return readings, metadata, adj

    def load(self):
        readings, metadata, adj = self.load_raw()
        # impute missing observations using last observed values
        # in authors' code: readings = readings.fillna(0)
        mask = ~readings.isna().values
        if self.imputation_mode == "nearest":
            readings = readings.ffill().bfill()
        elif self.imputation_mode == "zero":
            readings = readings.fillna(0)
        return readings, mask, metadata, adj

    def compute_similarity(self, method: str, **kwargs):
        if method == "precomputed":
            # load precomputed adjacency matrix based on road distance
            return self.adj

import itertools
from typing import List, Optional, Sequence

from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.synch_mode import HORIZON
from tsl.datasets.prototypes.mixin import MissingValuesMixin

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
        
class TrafAirSplitter(Splitter):
    def __init__(self,
                 val_len: int = None,
                 test_months: Sequence = (3, 6, 9, 12)):
        super(TrafAirSplitter, self).__init__()
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

class AirCross(DatetimeDataset):
    similarity_options = {"precomputed"}

    def __init__(self,
                 root: str = None,
                 test_months: Sequence = (3, 6, 9, 12),
                 years: Sequence = (),
                 imputation_mode: Literal["nearest", "zero", None] = "zero",
                 freq: str = "h",
                 include_exog: bool = False,
                 exog: str = 'humd'):
        # set root path
        self.root = root
        self.years = years
        self.imputation_mode = imputation_mode
        self.test_months = test_months
        self.include_exog = include_exog
        self.exog = exog

        assert imputation_mode in ["nearest", "zero", None]
        assert exog in ['traffic', 'temp', 'humd', None]

        # Set dataset frequency here to resample when loading
        if freq is not None:
            freq = to_pandas_freq(freq)
        self.freq = freq

        # load dataset
        readings, mask, adj, air_metadata, tra_metadata, modality = self.load()
        self.tra_metadata = tra_metadata
        self.air_metadata = air_metadata
        self.modality = modality
        covariates = {"adj": (adj, 'n n')}
        
        super().__init__(target=readings,
                         freq=freq,
                         mask=mask,
                         covariates=covariates,
                         similarity_score="precomputed",
                         temporal_aggregation="mean",
                         spatial_aggregation="mean",
                         default_splitting_method='trafair',
                         name='AirCross')

    def load_raw(self):
        # load sensors information
        air_metadata = pd.read_csv(os.path.join(self.root_dir, 'air_metadata.csv'))
        self.air_max_nodes = len(air_metadata)

        readings = pd.read_csv(os.path.join(self.root_dir, f'full_data_{self.exog}.csv'), index_col=0, parse_dates=['Time'])
        if len(self.years) != 0:
            readings = readings[readings.index.year.isin(self.years)]

        modality = np.zeros((len(readings.columns), 1))
        modality[self.air_max_nodes:] = 1

        if not self.include_exog:
            readings = readings.iloc[:, :self.air_max_nodes]
            modality = modality[:self.air_max_nodes]

        # resample here to aggregate only valid observations and
        # align to authors' preprocessing
        if self.freq is not None:
            readings = readings.apply(pd.to_numeric, errors='coerce')
            readings = readings.resample(self.freq).mean()

        # load adjacency
        ar_edge_index, ar_edge_weight = np.load(os.path.join(self.root_dir, 'air_adj.npz')).values()
        ar_adj = np.eye(self.air_max_nodes, dtype=np.float32)
        ar_adj[tuple(ar_edge_index)] = ar_edge_weight

        # Get adj for exogenous modality 
        tra_metadata = pd.DataFrame()
        if self.include_exog:
            tra_metadata = pd.read_csv(os.path.join(self.root_dir, f'{self.exog}_metadata.csv'))
            self.tra_max_nodes = len(tra_metadata)

            tr_edge_index, tr_edge_weight = np.load(os.path.join(self.root_dir, f'{self.exog}_adj.npz')).values()
            cr_edge_index, cr_edge_weight = np.load(os.path.join(self.root_dir, f'cross_adj_{self.exog}.npz')).values()
            # build square adj from coo to add adj as covariate

            tr_adj = np.eye(self.tra_max_nodes, dtype=np.float32)
            tr_adj[tuple(tr_edge_index)] = tr_edge_weight

            cr_adj = np.zeros((self.air_max_nodes, self.tra_max_nodes), dtype=np.float32)
            cr_adj[tuple(cr_edge_index)] = cr_edge_weight

            # cross_adj = pd.read_csv(os.path.join(self.root_dir, f'cross_dist_{self.exog}.csv'))
            adj = np.block([
                [ar_adj,  cr_adj],
                [cr_adj.T, tr_adj]
            ])
        else:
            adj = ar_adj

        return readings, adj, air_metadata, tra_metadata, modality
    
    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'trafair':
            val_len = kwargs.get('val_len')
            return TrafAirSplitter(test_months=self.test_months,
                                    val_len=val_len)

    def load(self):
        readings, adj, air_metadata, tra_metadata, modality = self.load_raw()
        # impute missing observations using last observed values
        # in authors' code: readings = readings.fillna(0)
        mask = ~readings.isna().values
        if self.imputation_mode == "nearest":
            readings = readings.ffill().bfill()
        elif self.imputation_mode == "zero":
            readings = readings.fillna(0)
        return readings, mask, adj, air_metadata, tra_metadata, modality

    def compute_similarity(self, method: str, **kwargs):
        if method == "precomputed":
            # load precomputed adjacency matrix based on road distance
            return self.adj

class StandardScalerSplit(Scaler):
    """Apply standardization to data by removing mean and scaling to unit
    variance.

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
    """

    def __init__(self, split: int, axis: Union[int, Tuple] = 0):
        super(StandardScalerSplit, self).__init__()
        self.axis = axis
        self.split = split

    @fit_wrapper
    def fit(self, x: TensArray, mask=None, keepdims=True):
        r"""Fit scaler's parameters `bias` :math:`\mu` and `scale`
        :math:`\sigma` as the mean and the standard deviation of :obj:`x`,
        respectively.

        Args:
            x: array-like input
            mask (optional): boolean mask to denote elements of :obj:`x` on
                which to fit the parameters.
                (default: :obj:`None`)
            keepdims (bool): whether to keep the same dimensions as :obj:`x` in
                the parameters.
                (default: :obj:`True`)
        """
        if mask is not None:
            x = np.where(mask, x, np.nan)
            t, n, f = x.shape

            first_half = x[:, :self.split, :] 
            second_half = x[:, self.split:, :]  

            first = np.nanmean(first_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)
            second = np.nanmean(second_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))

            self.bias = np.concatenate([filled_first, filled_second], axis=1)

            first = np.nanstd(first_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)
            second = np.nanstd(second_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))

            self.scale = np.concatenate([filled_first, filled_second], axis=1)
        else:
            t, n, f = x.shape

            first_half = x[:, :self.split, :] 
            second_half = x[:, self.split:, :]  

            first = first_half.mean(axis=(0, 1), keepdims=True)  # (1, 1, f)
            second = second_half.mean(axis=(0, 1), keepdims=True)  # (1, 1, f)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))
            
            self.bias = np.concatenate([filled_first, filled_second], axis=1)

            first = first_half.std(axis=(0, 1), keepdims=True)  # (1, 1, f)
            second = second_half.std(axis=(0, 1), keepdims=True)  # (1, 1, f)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))

            self.scale = np.concatenate([filled_first, filled_second], axis=1)
        self.scale = zeros_to_one_(self.scale)
        return self

    def transform(self, x: TensArray):
        r"""Apply transformation :math:`f(x) = (x - \mu) / \sigma`."""
        return (x - self.bias) / (self.scale + tsl.epsilon)

    def inverse_transform(self, x: TensArray):
        r"""Apply inverse transformation
        :math:`f(x) = (x \cdot \sigma) + \mu`."""
        return x * (self.scale + tsl.epsilon) + self.bias

    def fit_transform(self, x: TensArray, *args, **kwargs):
        r"""Fit scaler's parameters using input :obj:`x` and then transform
        :obj:`x`."""
        self.fit(x, *args, **kwargs)
        return self.transform(x)
    
class ScalerSplitModule(ScalerModule):
    def __init__(self,
                 scaler: Optional[Union["Scaler", "ScalerModule"]] = None,
                 *,
                 bias: Union[Tensor, float] = 0.,
                 scale: Union[Tensor, float] = 1.,
                 pattern: Optional[str] = None):
        super().__init__(scaler, bias=bias, scale=scale, pattern=pattern)
        self.bias_list = torch.unique(scaler.bias)
        self.scale_list = torch.unique(scaler.scale)

    def _get_name(self):
        return self.__class__.__name__

    def transform_tensor(self, x: Tensor, split = None) -> Tensor:
        if split:
            temp_bias = torch.zeros_like(x).to(x.device)
            temp_bias[:, :, :split] = self.bias_list[0]
            temp_bias[:, :, split:] = self.bias_list[1]

            temp_scale = torch.zeros_like(x).to(x.device)
            temp_scale[:, :, :split] = self.scale_list[0]
            temp_scale[:, :, split:] = self.scale_list[1]

            return (x - temp_bias) / temp_scale + tsl.epsilon
        else:
            return (x - self.bias) / self.scale + tsl.epsilon

    def inverse_transform_tensor(self, x: Tensor, split = None) -> Tensor:
        if split:
            temp_bias = torch.zeros_like(x).to(x.device)
            temp_bias[:, :, :split] = self.bias_list[0]
            temp_bias[:, :, split:] = self.bias_list[1]

            temp_scale = torch.zeros_like(x).to(x.device)
            temp_scale[:, :, :split] = self.scale_list[0]
            temp_scale[:, :, split:] = self.scale_list[1]

            return x * (temp_scale + tsl.epsilon) + temp_bias
        else:
            return x * (self.scale + tsl.epsilon) + self.bias

    def transform(self, x, split=None):
        split_trans_tensor = partial(self.transform_tensor, split=split)
        return recursive_apply(x, split_trans_tensor)

    def inverse_transform(self, x, split=None):
        split_invtr_tensor = partial(self.inverse_transform_tensor, split=split)
        return recursive_apply(x, split_invtr_tensor)
    
    def slice(self,
              time_index: Union[List, Tensor] = None,
              node_index: Union[List, Tensor] = None):
        if self.pattern is None:
            raise RuntimeError("You are trying to slice a scaler with no "
                               "pattern.")
        # move to new object
        scaler = ScalerSplitModule(self)
        # shortcut for when scaler is time-unvarying and node_index is None
        if time_index is None and node_index is None:
            return scaler

        # if time-unvarying scaler, just apply unsqueezing indexing
        new_axes, pattern = None, scaler.pattern
        if time_index is not None and time_index.ndim == 2:
            new_axes = torch.zeros(1, 1, dtype=torch.long)
            pattern = 'b ' + scaler.pattern

        # compute actual slicing for each param
        t, n = self.t_axis, self.n_axis  # axis of time and node dimensions
        ti_bias = ti_scale = time_index
        ni_bias = ni_scale = node_index
        if self.t_axis is not None:
            ti_bias = time_index if self.bias.size(t) > 1 else new_axes
            ti_scale = time_index if self.scale.size(t) > 1 else new_axes
        if self.n_axis is not None:
            ni_bias = node_index if self.bias.size(n) > 1 else None
            ni_scale = node_index if self.scale.size(n) > 1 else None

        # slice params
        scaler.bias = take(scaler.bias,
                           self.pattern,
                           time_index=ti_bias,
                           node_index=ni_bias)
        scaler.scale = take(scaler.scale,
                            self.pattern,
                            time_index=ti_scale,
                            node_index=ni_scale)
        # update pattern
        scaler.pattern = pattern

        return scaler

    # you can also override other methods if needed

class CrossSpatioTemporalDataset(ImputationDataset):
    def __init__(self,
                 target,
                 eval_mask,
                 index = None,
                 mask = None,
                 connectivity  = None,
                 covariates = None,
                 input_map = None,
                 target_map = None,
                 auxiliary_map = None,
                 scalers = None,
                 trend = None,
                 transform = None,
                 window: int = 12,
                 stride: int = 1,
                 window_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        # call parent constructor
        super().__init__(target=target,
                         eval_mask=eval_mask,
                         index=index,
                         mask=mask,
                         connectivity=connectivity,
                         covariates=covariates,
                         input_map=input_map,
                         target_map=target_map,
                         auxiliary_map=auxiliary_map,
                         scalers=scalers,
                         trend=trend,
                         transform=transform,
                         window=window,
                         stride=stride,
                         window_lag=window_lag,
                         precision=precision,
                         name=name)

    def expand_scaler(self, key: str, pattern: Optional[str] = None,
                      time_index: Union[List, Tensor] = None,
                      node_index: Union[List, Tensor] = None) \
            -> Optional[ScalerSplitModule]:
        # check if there is a scaler
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        elif key not in self.scalers:
            return None
        # convert indices
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_time_index(node_index, layout='index')
        # get params
        if pattern is None:
            return self.scalers[key]
        # if there is an out-pattern, create new scaler
        scaler = ScalerSplitModule(self.scalers[key], pattern=pattern)
        pattern = self.patterns[key] + ' -> ' + pattern
        scaler.bias = broadcast(scaler.bias,
                                pattern,
                                backend=torch,
                                time_index=time_index,
                                node_index=node_index)
        scaler.scale = broadcast(scaler.scale,
                                 pattern,
                                 backend=torch,
                                 time_index=time_index,
                                 node_index=node_index)
        return scaler

    def get_tensor(self, key: str, preprocess: bool = False,
                   time_index: Union[List, Tensor] = None,
                   node_index: Union[List, Tensor] = None) \
            -> Tuple[Tensor, Optional[ScalerSplitModule]]:
        # get dataset item
        if key not in self.keys:
            raise KeyError(f"{key} not in dataset {self.name}.")

        # convert indices
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_time_index(node_index, layout='index')
        x = take(getattr(self, key),
                 self.patterns[key],
                 backend=torch,
                 time_index=time_index,
                 node_index=node_index)
        try:
            test = x[:, :, :23, :]
        except:
            pass

        # get scaler (if any)
        scaler = None
        if key in self.scalers is not None:
            scaler = self.scalers[key].slice(time_index=time_index,
                                             node_index=node_index)
            if preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_item_elem(self, key: str,
                          time_index: Union[List, Tensor] = None,
                          node_index: Union[List, Tensor] = None) \
            -> Tuple[Tensor, Optional[ScalerSplitModule]]:
        # get batch item
        if key in self.input_map:
            itm = self.input_map[key]
        elif key in self.target_map:
            itm = self.target_map[key]
        else:
            raise KeyError(f"{key} not in any batch map of {self.name}.")

        # expand and concatenate tensors
        x = torch.cat([
            self.expand_tensor(k, itm.pattern, time_index, node_index)
            for k in itm.keys
        ],
                      dim=itm.cat_dim)

        # get scaler (if any)
        scaler = None
        if key in self._batch_scalers:
            scaler = self._batch_scalers[key].slice(time_index=time_index,
                                                    node_index=node_index)
            if itm.preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_keys(self,
                     keys: Iterable,
                     preprocess: bool = False,
                     time_index: Union[List, Tensor] = None,
                     node_index: Union[List, Tensor] = None,
                     cat_dim: Optional[int] = None,
                     return_pattern: bool = False):
        if any([key not in self.keys for key in keys]):
            unmatch = set(keys).difference(self.keys)
            raise KeyError(f"{unmatch} not in {self.name}.")
        pattern = outer_pattern([self.patterns[key] for key in keys])
        tensors, scalers = list(), list()
        for key in keys:
            tensor = self.expand_tensor(key, pattern, time_index, node_index)
            scaler = self.expand_scaler(key, pattern, time_index, node_index)
            if preprocess and scaler is not None:
                tensor = scaler(tensor)
            tensors.append(tensor)
            scalers.append(scaler)
        if len(tensors) == 1:
            if return_pattern:
                return tensors[0], scalers[0], pattern
            return tensors[0], scalers[0]
        if cat_dim is not None:
            scalers = ScalerSplitModule.cat(scalers,
                                       dim=cat_dim,
                                       sizes=[t.size() for t in tensors])
            tensors = torch.cat(tensors, dim=cat_dim)
        if return_pattern:
            return tensors, scalers, pattern
        return tensors, scalers

    def get_mask(self, dtype: Union[type, str, np.dtype] = None) -> Tensor:
        mask = self.mask if self.has_mask else ~torch.isnan(self.target)
        if dtype is not None:
            assert dtype in ['bool', 'uint8', bool, torch.bool, torch.uint8]
            mask = mask.to(dtype)
        return mask

    def add_scaler(self, key: str, scaler: Union[Scaler, ScalerSplitModule]):
        r"""Add a :class:`tsl.data.preprocessing.Scaler` for the object indexed
        by :obj:`key` in the dataset.

        Args:
            key (str): The name of the variable associated to the scaler. It
                must be a temporal variable, i.e., :obj:`data` or an exogenous.
            scaler (Scaler): The :class:`~tsl.data.preprocessing.Scaler`.
        """
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        # copy to ScalerModule
        scaler = ScalerSplitModule(scaler)
        pattern = self.patterns[key]
        self._check_pattern(scaler.bias,
                            pattern,
                            name=f"scaler ({key})",
                            allow_broadcasting=True)
        self._check_pattern(scaler.scale,
                            pattern,
                            name=f"scaler ({key})",
                            allow_broadcasting=True)
        if key == 'target' and self.trend is not None:
            self.__target_bias = scaler.bias
            scaler.bias = scaler.bias + self.trend
        scaler.pattern = pattern
        self.scalers[key] = scaler
        # cache batch scaler if target tensor is in a multi-key batch item
        for bm in [self.input_map, self.target_map, self.auxiliary_map]:
            for bm_key, bm_item in bm.items():
                if key in bm_item.keys and len(bm_item.keys) > 1:
                    tensor, scaler = self.collate_keys(bm_item.keys,
                                                       cat_dim=bm_item.cat_dim,
                                                       return_pattern=False)
                    self._batch_scalers[bm_key] = scaler


from typing import Literal, Mapping, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torch import Generator

import tsl

from tsl.typing import Index
from tsl.data.loader import StaticGraphLoader
from tsl.data.spatiotemporal_dataset import SpatioTemporalDataset
from tsl.data.datamodule.splitters import Splitter

StageOptions = Literal['fit', 'validate', 'test', 'predict']


class SpatioTemporalDataModule(LightningDataModule):
    r"""Base :class:`~pytorch_lightning.core.LightningDataModule` for
    :class:`~tsl.data.SpatioTemporalDataset`.

    Args:
        dataset (SpatioTemporalDataset): The complete dataset.
        scalers (dict, optional): Named mapping of
            :class:`~tsl.data.preprocessing.scalers.Scaler`
            to be used for data rescaling after splitting. Every scaler is given
            as input the attribute of the dataset named as the scaler's key.
            If :obj:`None`, no scaling is performed.
            (default :obj:`None`)
        mask_scaling (bool): If :obj:`True`, then compute statistics for
            :obj:`dataset.target` scaler (if any) by considering only valid
            values (according to :obj:`dataset.mask`).
            (default :obj:`True`)
        splitter (Splitter, optional): The
            :class:`~tsl.data.datamodule.splitters.Splitter` to be used for
            splitting :obj:`dataset` into train/validation/test sets.
            (default :obj:`None`)
        batch_size (int): Size of the mini-batches for the dataloaders.
            (default :obj:`32`)
        workers (int): Number of workers to use in the dataloaders.
            (default :obj:`0`)
        pin_memory (bool): If :obj:`True`, then enable pinned GPU memory for
            :meth:`~tsl.data.datamodule.SpatioTemporalDataModule.train_dataloader`.
            (default :obj:`False`)
    """

    def __init__(self,
                 dataset: SpatioTemporalDataset,
                 scalers: Optional[Mapping] = None,
                 mask_scaling: bool = True,
                 splitter: Optional[Splitter] = None,
                 batch_size: int = 32,
                 workers: int = 0,
                 pin_memory: bool = False,
                 generator: Generator = None):
        super(SpatioTemporalDataModule, self).__init__()
        self.torch_dataset = dataset
        # splitting
        self.splitter = splitter
        self.trainset = self.valset = self.testset = None
        self.generator = generator
        # scaling
        if scalers is None:
            self.scalers = dict()
        else:
            self.scalers = scalers
        self.mask_scaling = mask_scaling
        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

    def __getattr__(self, item):
        ds = self.__dict__.get('torch_dataset')
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def __repr__(self):
        return "{}(train_len={}, val_len={}, test_len={}, " \
               "scalers=[{}], batch_size={})" \
            .format(self.__class__.__name__,
                    self.train_len, self.val_len, self.test_len,
                    ', '.join(self.scalers.keys()), self.batch_size)

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def testset(self):
        return self._testset

    @trainset.setter
    def trainset(self, value):
        self._add_set('train', value)

    @valset.setter
    def valset(self, value):
        self._add_set('val', value)

    @testset.setter
    def testset(self, value):
        self._add_set('test', value)

    @property
    def train_len(self):
        return len(self.trainset) if self.trainset is not None else None

    @property
    def val_len(self):
        return len(self.valset) if self.valset is not None else None

    @property
    def test_len(self):
        return len(self.testset) if self.testset is not None else None

    @property
    def train_slice(self):
        return self._train_slice if hasattr(self, '_train_slice') else None

    @property
    def val_slice(self):
        return self._val_slice if hasattr(self, '_val_slice') else None

    @property
    def test_slice(self):
        return self._test_slice if hasattr(self, '_test_slice') else None

    def _add_set(self, split_type, _set):
        assert split_type in ['train', 'val', 'test']
        split_type = '_' + split_type
        name = split_type + 'set'
        if _set is None or isinstance(_set, Dataset):
            setattr(self, name, _set)
        else:
            indices = _set
            assert isinstance(indices, Index.__args__), \
                f"type {type(indices)} of `{name}` is not a valid type. " \
                "It must be a dataset or a sequence of indices."
            _set = Subset(self.torch_dataset, indices)
            _slice = self.torch_dataset.expand_indices(_set.indices,
                                                       merge=True)
            setattr(self, name, _set)
            slice_name = split_type + '_slice'  # e.g. trainset > _train_slice
            setattr(self, slice_name, _slice)

    def setup(self, stage: StageOptions = None):
        # splitting
        if self.splitter is not None:
            self.splitter.split(self.torch_dataset)
            self.trainset = self.splitter.train_idxs
            self.valset = self.splitter.val_idxs
            self.testset = self.splitter.test_idxs

        for key, scaler, in self.scalers.items():
            if key not in self.torch_dataset:
                raise RuntimeError("Cannot find a tensor to scale matching "
                                   f"key '{key}'.")
            # set scalers
            if stage == 'predict':
                tsl.logger.info(f'Set scaler for {key}: {scaler}')
            else:  # fit scalers before training
                data = getattr(self.torch_dataset, key)
                # get only training slice
                if 't' in self.torch_dataset.patterns[key]:
                    data = data[self.train_slice]

                mask = None
                if key == 'target' and self.mask_scaling:
                    if self.torch_dataset.mask is not None:
                        mask = self.torch_dataset.get_mask()[self.train_slice]

                scaler = scaler.fit(data, mask=mask, keepdims=True)
                tsl.logger.info(f'Fit and set scaler for {key}: {scaler}')
            self.torch_dataset.add_scaler(key, scaler)

    def get_dataloader(self, split: Literal['train', 'val', 'test'] = None,
                       shuffle: bool = False,
                       batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        if split is None:
            dataset = self.torch_dataset
        elif split in ['train', 'val', 'test']:
            dataset = getattr(self, f'{split}set')
        else:
            raise ValueError("Argument `split` must be one of "
                             "'train', 'val', or 'test'.")
        if dataset is None:
            return None
        pin_memory = self.pin_memory if split == 'train' else None
        return StaticGraphLoader(dataset,
                                 batch_size=batch_size or self.batch_size,
                                 shuffle=shuffle,
                                 drop_last=split == 'train',
                                 num_workers=self.workers,
                                 pin_memory=pin_memory,
                                 generator=self.generator)

    def train_dataloader(self, shuffle: bool = True,
                         batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('train', shuffle, batch_size)

    def val_dataloader(self, shuffle: bool = False,
                       batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('val', shuffle, batch_size)

    def test_dataloader(self, shuffle: bool = False,
                        batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('test', shuffle, batch_size)

import math
from typing import Mapping, Type

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from tsl.datasets import TabularDataset
from tsl.ops.connectivity import parse_connectivity
from tsl.typing import SparseTensArray
from tsl.utils.casting import torch_to_numpy
from tsl.utils.python_utils import foo_signature

from torch_geometric.utils import dense_to_sparse


class CrossGaussianNoiseSyntheticDataset(TabularDataset):
    r"""A generator of synthetic datasets from an input model and input graph.

    The input model must be implemented as a :class:`torch.nn.Module` and must
    return the observation at the next step and (optionally) the hidden state
    for the next step. Gaussian noise will be added to the output of the model
    at each step.

    Args:
        num_features (int): Number of features in the generated dataset.
        num_nodes (int): Number of nodes in the graph.
        num_steps (int): Number of steps to generate.
        connectivity (SparseTensArray): Connectivity of the underlying graph.
        model (torch.nn.Module): Model used to generate data. If :obj:`None`,
            it will attempt to create model from ``model_class`` and
            ``model_kwargs``.
        model_class (type, optional): Class of the model used to generate the
            data.
            (default: :obj:`None`)
        model_kwargs (dict, optional): Keyword arguments needed to initialize
            the model.
            (default: :obj:`None`)
        sigma_noise (float): Standard deviation of the noise.
            (default: :obj:`0.2`)
        name (str, optional): Name for the generated dataset.
            (default: :obj:`None`)
        seed (int, optional): Seed for the random number generator.
            (default: :obj:`None`)
    """

    seed: int = None

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 split: int,
                 num_steps: int,
                 connectivity: SparseTensArray,
                 min_window: int = 1,
                 o_model: nn.Module = None,
                 o_model_class: Type = None,
                 o_model_kwargs: Mapping = None,
                 o_sigma_noise: float = .2,
                 e_model: nn.Module = None,
                 e_model_class: Type = None,
                 e_model_kwargs: Mapping = None,
                 e_sigma_noise: float = .2,
                 include_exog: bool = True,
                 name: str = None,
                 seed: int = 42,
                 **kwargs):
        self.name = name
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._num_steps = num_steps
        self._min_window = min_window
        self._include_exog = include_exog
        if seed is not None:
            self.seed = seed

        if o_model is not None:
            self.o_model = o_model
        else:
            self.o_model = o_model_class(**o_model_kwargs)

        self._model_forward_signature = foo_signature(o_model.forward)

        self.o_sigma_noise = o_sigma_noise

        if e_model is not None:
            self.e_model = e_model
        else:
            self.e_model = e_model_class(**e_model_kwargs)

        self._model_forward_signature = foo_signature(e_model.forward)

        self.e_sigma_noise = e_sigma_noise

        if connectivity is not None:
            self.connectivity = parse_connectivity(connectivity,
                                                   target_layout='edge_index',
                                                   num_nodes=num_nodes)
        else:
            self.connectivity = None
        self._main_num = split
        self._exog_num = connectivity.shape[1] - split

        target, optimal_pred, mask, modality = self.load()
        self.modality = modality
        super().__init__(target=target, mask=mask, name=name, **kwargs)

        self.add_covariate('optimal_pred', optimal_pred, 't n f')

    def load_raw(self, *args, **kwargs):
        return self.generate_data(self.seed)

    # @property
    # def mae_optimal_model(self):
    #     r""":math:`\mathbb{E}[|\mathbf{X}|]` of a Gaussian
    #     :math:`\mathbf{X} \sim \mathcal{N}(0, \sigma^2)`, computed as
    #     :math:`\varepsilon = \sqrt{\frac{2}{\pi}}\sigma`.
    #     """
    #     return math.sqrt(2.0 / math.pi) * self.o_sigma_noise

    def _filter_forward_kwargs(self, kwargs):
        if not self._model_forward_signature['has_kwargs']:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in self._model_forward_signature['signature']
            }
        return kwargs

    def _model_forward(self, *args, **kwargs):
        kwargs = self._filter_forward_kwargs(kwargs)
        out = self.o_model(*args, **kwargs)
        if len(out) != 2:
            return out, None
        # Assumes that if the output has length 2,
        # then it will contain [output, hidden_state].
        return out

    def _e_model_forward(self, *args, **kwargs):
        kwargs = self._filter_forward_kwargs(kwargs)
        out = self.e_model(*args, **kwargs)
        if len(out) != 2:
            return out, None
        # Assumes that if the output has length 2,
        # then it will contain [output, hidden_state].
        return out

    def generate_data(self, seed=None):
        """"""
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # initialize with noise
        x = torch.empty(
            (self._num_steps + self._min_window, self._num_nodes,
             self._num_features)).normal_(generator=rng) * self.o_sigma_noise

        y_opt = torch.empty(
            (self._num_steps, self._num_nodes, self._num_features))

        if self.connectivity is None:
            edge_index = edge_weight = None
        else:
            edge_index, edge_weight = self.connectivity

            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1])

            adj = torch.eye(self._num_nodes, dtype=torch.float32)
            adj[tuple(edge_index)] = edge_weight

            o_edge_index, o_edge_weight = dense_to_sparse(adj[:self._main_num, :self._main_num]) # N N
            e_edge_index, e_edge_weight = dense_to_sparse(adj[self._main_num:, self._main_num:]) # M M

            c_adj = adj[:self._main_num, self._main_num:]
            # c_edge_index, c_edge_weight = dense_to_sparse(c_adj) # N M

        with torch.no_grad():
            eh_t = None
            oh_t = None
            for t in tqdm(range(self._min_window,
                                self._min_window + self._num_steps),
                          desc=f"Generating {self.__class__.__name__} data"):
                # ft modelling 
                e_t, eh_t = self._e_model_forward(x[None, t - self._min_window:t, self._main_num:],
                                               h=eh_t,
                                               t=t,
                                               edge_index=e_edge_index,
                                               edge_weight=e_edge_weight)
                f_t = e_t + torch.zeros_like(e_t).normal_(generator=rng) * self.e_sigma_noise
                x[t:t + 1, self._main_num:] = f_t[0]

                Uf_t = c_adj @ f_t

                # Adding to original graph
                o_t, oh_t = self._model_forward(x[None, t - self._min_window:t, :self._main_num],
                                               h=oh_t,
                                               t=t,
                                               edge_index=o_edge_index,
                                               edge_weight=o_edge_weight)
                x_t = torch.tanh(o_t + Uf_t)
                
                y_opt[t - self._min_window:t + 1 - self._min_window, :self._main_num] = x_t[0]
                # add noise
                x_t = x_t + torch.zeros_like(x_t).normal_(
                    generator=rng) * self.o_sigma_noise
                x[t:t + 1, :self._main_num] = x_t[0]

        x = torch_to_numpy(x[self._min_window:])
        y_opt = torch_to_numpy(y_opt)

        modality = np.zeros((self._num_nodes, 1))
        modality[self._main_num:] = 1

        # Just take the original graph if not including exogeneous data
        if not self._include_exog:
            if self.connectivity is not None:
                self.connectivity = parse_connectivity(o_edge_index,
                                                    target_layout='edge_index',
                                                    num_nodes=self._main_num)
            else:
                self.connectivity = None
            
            x = x[:, :self._main_num]
            y_opt = y_opt[:, :self._main_num]

        return x, y_opt, np.ones_like(x), modality

    def get_connectivity(self, layout: str = 'edge_index', **kwargs):
        """"""
        if self.connectivity is not None:
            return parse_connectivity(connectivity=self.connectivity,
                                      target_layout=layout,
                                      num_nodes=self.n_nodes)
        return None

from typing import List, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.utils import add_self_loops

from tsl.nn.layers.graph_convs.gpvar import GraphPolyVAR
from tsl.ops.graph_generators import build_tri_community_graph

class _GPVAR(GraphPolyVAR):
    def forward(self, x, edge_index, edge_weight=None):
        out = super(_GPVAR, self).forward(x, edge_index, edge_weight)
        return torch.tanh(out)

SIZES_X = [10, 10, 10, 10]
PROB_X = [[0.30, 0.01, 0.01, 0.01],
          [0.01, 0.30, 0.01, 0.01],
          [0.01, 0.01, 0.30, 0.01],
          [0.01, 0.01, 0.01, 0.30]]

SIZES_Y = [15, 15, 15]
PROB_Y = [[0.30, 0.01, 0.01],
          [0.01, 0.30, 0.01],
          [0.01, 0.01, 0.30]]

# Cross-layer bipartite SBM
# Y has 2 blocks, X has 2 blocks
SIZES_XY = ([15, 15, 15], [10, 10, 10, 10])
PROB_XY = [[0.25, 0.00, 0.00],
           [0.10, 0.10, 0.00],
           [0.00, 0.10, 0.10],
           [0.00, 0.00, 0.25]]

DEFAULT_SBM_PARAMS = {'sizes_x': SIZES_X, 'prob_x': PROB_X,
                      'sizes_y': SIZES_Y, 'prob_y': PROB_Y,
                      'sizes_xy': SIZES_XY, 'prob_xy': PROB_XY}

class CrossGPVARDataset(CrossGaussianNoiseSyntheticDataset):
    """Generator for synthetic datasets from a graph polynomial VAR filter on
    triangular community graphs as shown in the paper `"AZ-whiteness test: a
    test for uncorrelated noise on spatio-temporal graphs"
    <https://arxiv.org/abs/2204.11135>`_ (Zambon et al., NeurIPS 22).

    Args:
        num_communities (int): Number of communities (triangles) in the graph.
        num_steps (int): Length of the generated sequence.
        filter_params (iterable): Parameters of the graph polynomial filter
            used to generate the dataset.
        sigma_noise (float): Standard deviation of the noise.
        norm (str): The normalization used for edges and edge weights. The
            available options are: :obj:`'gcn'`, :obj:`'asym'` and
            :obj:`'none'`.
            (default: :obj:`'none'`)
        name (optional, str): Name of the dataset.
    """

    def __init__(self,
                 num_steps: int,
                 o_filter_params: Union[List, Tensor, ndarray],
                 e_filter_params: Union[List, Tensor, ndarray],
                 o_sigma_noise: float = .2,
                 o_norm: str = 'none',
                 e_sigma_noise: float = .2,
                 e_norm: str = 'none',
                 sbm_params: dict = DEFAULT_SBM_PARAMS,
                 include_exog: bool = True,
                 name: str = None):
        if name is None:
            name = "GP-VAR"

        # TODO: Change the graph generation process 
        # node_idx, edge_index, _ = build_tri_community_graph(
        #     num_communities=num_communities)
        # num_nodes = len(node_idx)
        # # add self loops
        # edge_index, _ = add_self_loops(edge_index=torch.tensor(edge_index),
        #                                num_nodes=num_nodes)
        # split = 5

        Gx, Gy, Gxy, A_aug = generate_multiplex_sbm(
            **sbm_params,
            seed=42
        )
        num_nodes = A_aug.shape[0]
        self.air_max_nodes = len(Gx.nodes)

        edge_index = []
        edge_weight = []

        for i, row in enumerate(A_aug):
            for j, el in enumerate(row):
                if el > 0 and i != j and np.isfinite(el):
                    edge_index.append([i, j])
                    edge_weight.append(el)

        edge_index = np.array(edge_index).T
        edge_weight = np.array(edge_weight)

        # Calculate the filters
        if not isinstance(o_filter_params, Tensor):
            o_filter_params = torch.as_tensor(o_filter_params, dtype=torch.float32)
        if not isinstance(e_filter_params, Tensor):
            e_filter_params = torch.as_tensor(e_filter_params, dtype=torch.float32)

        o_filter = _GPVAR.from_params(filter_params=o_filter_params,
                                    norm=o_norm,
                                    cached=True)
        e_filter = _GPVAR.from_params(filter_params=e_filter_params,
                                    norm=e_norm,
                                    cached=True)  

        super(CrossGPVARDataset, self).__init__(num_features=1,
                                           num_nodes=num_nodes,
                                           num_steps=num_steps,
                                           split=self.air_max_nodes,
                                           connectivity=edge_index,
                                           min_window=o_filter.temporal_order,
                                           o_model=o_filter,
                                           o_sigma_noise=o_sigma_noise,
                                           e_model=e_filter,
                                           e_sigma_noise=e_sigma_noise,
                                           include_exog=include_exog,
                                           name=name)
        

import numpy as np
import networkx as nx

def generate_multiplex_sbm(
    sizes_x,
    prob_x,
    sizes_y,
    prob_y,
    sizes_xy,
    prob_xy,
    seed=None
):
    """
    Generate a multiplex graph with:
    - Layer X: SBM(sizes_x, prob_x)  → placed in UPPER block
    - Layer Y: SBM(sizes_y, prob_y)  → placed in LOWER block
    - Cross-layer edges via bipartite SBM(sizes_xy, prob_xy)

    Returns:
        Gx, Gy, Gxy, A_aug
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1. Layer X (SBM) – goes to upper-left
    # -----------------------------
    Gx = nx.stochastic_block_model(
        sizes_x, prob_x, seed=seed
    )
    n_x = sum(sizes_x)

    # -----------------------------
    # 2. Layer Y (SBM) – goes to lower-right
    # -----------------------------
    Gy = nx.stochastic_block_model(
        sizes_y, prob_y, seed=(seed + 1 if seed else None)
    )
    n_y = sum(sizes_y)

    # -----------------------------
    # 3. Cross-layer bipartite SBM
    # -----------------------------
    sizes_y_xy, sizes_x_xy = sizes_xy
    B_xy = prob_xy

    Gxy = nx.Graph()

    # X nodes: block labels
    x_blocks = []
    for b, size in enumerate(sizes_x_xy):
        x_blocks += [b] * size

    # Y nodes: block labels
    y_blocks = []
    for b, size in enumerate(sizes_y_xy):
        y_blocks += [b] * size

    # Add bipartite node sets:
    #  - X nodes: 0 .. n_x - 1  (bipartite = 0)
    #  - Y nodes: n_x .. n_x+n_y-1 (bipartite = 1)

    for i in range(n_x):
        Gxy.add_node(i, bipartite=0, block=x_blocks[i])

    for j in range(n_y):
        Gxy.add_node(j + n_x, bipartite=1, block=y_blocks[j])

    # Sample bipartite edges
    for i in range(n_x):
        for j in range(n_y):
            bi = x_blocks[i]
            bj = y_blocks[j]
            if rng.random() < B_xy[bi][bj]:
                Gxy.add_edge(i, j + n_x)

    # -----------------------------
    # 4. Build supra-adjacency
    # -----------------------------
    A_aug = np.zeros((n_x + n_y, n_x + n_y))

    # Layer X (upper-left)
    Ax = nx.to_numpy_array(Gx)
    A_aug[:n_x, :n_x] = Ax

    # Layer Y (lower-right)
    Ay = nx.to_numpy_array(Gy)
    A_aug[n_x:, n_x:] = Ay

    # Cross-layer adjacency
    Axy = nx.to_numpy_array(Gxy)

    # X → Y block (upper-right)
    A_aug[:n_x, n_x:] = Axy[:n_x, n_x:]

    # Y → X block (lower-left)
    A_aug[n_x:, :n_x] = Axy[n_x:, :n_x]

    return Gx, Gy, Gxy, A_aug

def add_missing_sensors_cross(dataset: AirCross | CrossGPVARDataset,
                              p_noise=0.05,
                              p_fault=0.01,
                              min_seq=1,
                              max_seq=10,
                              seed=None,
                              inplace=True,
                              masked_sensors = [],
                              connect = None,
                              spatial_shift = False, 
                              order = 0,
                              node_features = 'CC',
                              mode='road'):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.air_max_nodes, dataset.n_channels)
    adj = dataset.get_connectivity(**connect, layout='dense')
    air_adj = adj[:dataset.air_max_nodes, :dataset.air_max_nodes]  
    eval_mask = np.zeros_like(dataset.mask)

    if masked_sensors is None:
        if spatial_shift:
            tmp_mask = shift_mask(shape, feature=node_features, order=order, 
                                   adj=air_adj, p_noise=p_noise)
            dataset.seed = seed
        else:
            tmp_mask = sample_mask(shape,
                                    p=p_fault,
                                    p_noise=p_noise,
                                    mode=mode,
                                    adj=air_adj)
            
            dataset.p_fault = p_fault
            dataset.p_noise = p_noise
            dataset.min_seq = min_seq
            dataset.max_seq = max_seq
            dataset.seed = seed
            dataset.random = random

        # mask = rearrange(eval_mask, "b n 1 -> b n")
        mask_sum = tmp_mask.sum(0)  # n
        masked_sensors = (np.where(mask_sum > 0)[0]).tolist()
        eval_mask[:, :dataset.air_max_nodes] = tmp_mask
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