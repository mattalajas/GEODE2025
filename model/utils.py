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
