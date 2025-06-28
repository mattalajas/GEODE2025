import argparse
import torch
import math
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import numpy as np

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PeMS07, PvUS, LargeST, ElectricityBenchmark, PeMS04, Elergone, PemsBay
from tsl.datasets.prototypes import casting
from tsl.engines import Imputer
from tsl.experiment import Experiment
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import (BiRNNImputerModel, GRINModel, RNNImputerModel,
                           SPINHierarchicalModel, SPINModel)
from tsl.ops import similarities as sims
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

from my_datasets import AirQualitySmaller, AirQualityAuckland, AirQualityKrig, add_missing_sensors
from baselines.KITS import KITS
from baselines.IGNNK import IGNNK
from baselines.DIDA import DGNN
from baselines.CauSTG import CauSTG
from baselines.IGNNK import IGNNK
from baselines.LSJSTN import LSJSTN
from fillers.KITS_filler import GCNCycVirtualFiller
from fillers.unnamed_filler import UnnamedKrigFiller
from fillers.unnamed_filler_v2 import UnnamedKrigFillerV2
from fillers.unnamed_filler_v4 import UnnamedKrigFillerV4
from fillers.unnamed_filler_v5 import UnnamedKrigFillerV5
from fillers.DIDA_filler import DidaFiller
from fillers.CauSTG_filler import CauSTGFiller
from fillers.IGNNK_filler import IGNNKFiller
from fillers.LSJSTN_filler import LSJSTNFiller
from unnamedKrig import UnnamedKrigModel
from unnamedKrig_v2 import UnnamedKrigModelV2
from unnamedKrig_v3 import UnnamedKrigModelV3
from unnamedKrig_v4 import UnnamedKrigModelV4
from unnamedKrig_v5 import UnnamedKrigModelV5
from utils import month_splitter, test_wise_eval

MODELS = ['kits', 'unkrig', 'kcn', 'unkrigv2', 'unkrigv3', 'unkrigv4', 'unkrigv5', 'ignnk', 'lsjstn', 'caustg', 'dida']

def get_model_class(model_str):
    if model_str == 'rnni':
        model = RNNImputerModel
    elif model_str == 'birnni':
        model = BiRNNImputerModel
    elif model_str == 'grin':
        model = GRINModel
    elif model_str == 'spin':
        model = SPINModel
    elif model_str == 'spin-h':
        model = SPINHierarchicalModel
    elif model_str == 'kits':
        model = KITS
    elif model_str == 'unkrig':
        model = UnnamedKrigModel
    elif model_str == 'unkrigv2':
        model = UnnamedKrigModelV2
    elif model_str == 'unkrigv3':
        model = UnnamedKrigModelV3
    elif model_str == 'unkrigv4':
        model = UnnamedKrigModelV4
    elif model_str == 'unkrigv5':
        model = UnnamedKrigModelV5
    elif model_str == 'ignnk':
        model = IGNNK
    elif model_str == 'dida':
        model = DGNN
    elif model_str == 'caustg':
        model = CauSTG
    elif model_str == 'lsjstn':
        model = LSJSTN
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name: str, p_fault=0., p_noise=0., t_range = ['2022-04-01', '2022-12-01'],
                masked_s=None, agg_func = 'mean', test_month=[5], location='Auckland', connectivity=None, mode='road',
                spatial_shift=False, order=0, node_features='c_centrality'):
    if dataset_name == 'air':
        return AirQualityKrig(impute_nans=True, small=True, masked_sensors=masked_s, p=p_noise), masked_s
    if dataset_name == 'air_smaller':
        return AirQualitySmaller('data', impute_nans=True, masked_sensors=masked_s), masked_s
    if dataset_name == 'air_auckland' or dataset_name == 'air_invercargill1' or dataset_name == 'air_invercargill2':
        air_data = AirQualityAuckland('data', t_range=t_range, masked_sensors=masked_s, 
                                  agg_func=agg_func, test_months=test_month,
                                  location=location, p=p_noise)
        
        return add_missing_sensors(air_data,
                            p_fault=p_fault,
                            p_noise=p_noise,
                            min_seq=12,
                            max_seq=12 * 4, 
                            masked_sensors=masked_s,
                            connect=connectivity,
                            mode=mode,
                            spatial_shift=spatial_shift,
                            order=order,
                            node_features=node_features)
        

    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    if dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]

    if dataset_name == 'aqi':
        return add_missing_sensors(AirQuality(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4, 
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)

    if dataset_name == 'aqism':
        return add_missing_sensors(AirQuality(small=True),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4, 
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    
    if dataset_name == 'metrla':
        return add_missing_sensors(MetrLA(freq='5T'),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4, 
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'pem07':
        pems = PeMS07()

        masks = np.ones((pems.target.shape[0], pems.target.shape[1], 1))
        pems.set_mask(masks)
        
        return add_missing_sensors(pems,
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'pemsbay':
        pems = PemsBay()

        masks = np.ones((pems.target.shape[0], pems.target.shape[1], 1))
        pems.set_mask(masks)
        
        return add_missing_sensors(pems,
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    
    if dataset_name == 'pem04':
        pems = PeMS04()

        masks = np.ones((pems.target.shape[0], pems.target.shape[1], 1))
        pems.set_mask(masks)
        
        return add_missing_sensors(pems,
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)

    if dataset_name == 'nrel-al':
        pv_us = PvUS(zones='east')
        pv_us.metadata = pv_us.metadata[:137]
        cols = pv_us.target.columns[:137]
        pv_us.target = pv_us.target.loc[:, cols]

        masks = np.ones((pv_us.target.shape[0], pv_us.target.shape[1], 1))
        pv_us.set_mask(masks)
        
        return add_missing_sensors(pv_us,
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)

    if dataset_name == 'nrel-md':
        pv_us = PvUS(zones='east')
        pv_us.metadata = pv_us.metadata[1746: 1826]
        cols = pv_us.target.columns[1746: 1826]
        pv_us.target = pv_us.target.loc[:, cols]

        masks = np.ones((pv_us.target.shape[0], pv_us.target.shape[1], 1))
        pv_us.set_mask(masks)
        
        return add_missing_sensors(pv_us,
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    
    if dataset_name == 'sd':
        return add_missing_sensors(LargeST(subset='SD', year=[2019, 2020]),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)

    if dataset_name == 'electricity':
        df = ElectricityBenchmark()

        df.similarity_options = 'precomputed'
        train_df = df.dataframe()
        x = np.asarray(train_df) * df.mask[..., -1]
        period = casting.to_pandas_freq('1D').nanos // df.freq.nanos
        x = (x - x.mean()) / x.std()

        sim = sims.correntropy(x, period=period, mask=df.mask, gamma=10)

        def compute_similarity(self):
            return sim
        df.compute_similarity = compute_similarity
        
        return add_missing_sensors(df,
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  mode=mode,
                                  )

    raise ValueError(f"Dataset {dataset_name} not available in this setting.")

def run_imputation(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    torch.set_float32_matmul_precision('high')
    assert cfg.eval_setting in ['train_wise', 'test_wise']

    dataset, masked_sensors = get_dataset(cfg.dataset.name,
                            p_fault=cfg.dataset.get('p_fault'),
                            p_noise=cfg.dataset.get('p_noise'),
                            t_range=cfg.dataset.get('t_range'),
                            masked_s=cfg.dataset.get('masked_sensors'),
                            agg_func=cfg.dataset.get('agg_func'),
                            test_month=cfg.dataset.get('test_month'),
                            location=cfg.dataset.get('location'),
                            connectivity=cfg.dataset.get('connectivity'),
                            mode=cfg.dataset.get('mode'),
                            spatial_shift=cfg.dataset.get('spatial_shift'),
                            order=cfg.dataset.get('order'),
                            node_features=cfg.dataset.get('node_features'))

    print(f'Masked sensors: {masked_sensors}')

    # encode time of the day and use it as exogenous variable
    # covariates = {'u': dataset.datetime_encoded('day').values}

    # get adjacency matrix
    if cfg.model.name in MODELS:
        adj = dataset.get_connectivity(**cfg.dataset.connectivity, layout='dense')
    else:
        adj = dataset.get_connectivity(**cfg.dataset.connectivity)
    # u = np.expand_dims(covariates['u'], axis=1)
    # covariates['u'] = np.repeat(u, max(adj[0]), axis=1)
    # print(covariates['u'].shape)

    # instantiate dataset

    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                    #   covariates=covariates,
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=cfg.window,
                                      stride=cfg.stride)

    scalers = {'target': StandardScaler(axis=(0, 1))}

    if cfg.dataset.get('shift', False):
        dataset.get_splitter = month_splitter

        dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=dataset.get_splitter(**cfg.dataset.splitting),
            batch_size=cfg.batch_size,
            workers=cfg.workers)
    else:
        val_len = cfg.dataset.splitting.get('val_len')
        test_len = cfg.dataset.splitting.get('test_len')
        dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=dataset.get_splitter(val_len=val_len, test_len=test_len),
            batch_size=cfg.batch_size,
            workers=cfg.workers)
    dm.setup(stage='fit')

    print(f'train_times: {np.unique(dataset.dataframe().iloc[dm.train_dataloader().dataset.indices].index.month)}, \
          test_times: {np.unique(dataset.dataframe().iloc[dm.test_dataloader().dataset.indices].index.month)}')

    if cfg.get('in_sample', False):
        dm.trainset = list(range(len(torch_dataset)))

    ########################################
    # imputer                              #
    ########################################

    model_cls = get_model_class(cfg.model.name)
    
    if cfg.model.name == 'kits':
        model_kwargs = dict(adj=adj, d_in=dm.n_channels, n_nodes=dm.n_nodes, args=cfg.model)
    elif cfg.model.name == 'unkrig':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    elif cfg.model.name == 'unkrigv2':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    elif cfg.model.name == 'unkrigv3':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    elif cfg.model.name == 'unkrigv4':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    elif cfg.model.name == 'unkrigv5':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    elif cfg.model.name == 'ignnk':
        model_kwargs = dict(h=cfg.window)
    elif cfg.model.name == 'dida':
        model_kwargs = dict(nfeat=dm.n_channels, output_size=dm.n_channels, 
                            num_nodes=adj.shape[0], args=cfg.model.hparams)
    elif cfg.model.name == 'caustg':
        model_kwargs = dict(in_dim=dm.n_channels, out_dim=cfg.window, args=cfg.model.hparams)
    elif cfg.model.name == 'lsjstn':
        model_kwargs = dict(in_dim=dm.n_channels)
    elif cfg.model.name == 'kcn':
        pass
    else:
        model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                            input_size=torch_dataset.n_channels)

    model_cls.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    if cfg.model.name == 'kcn':
        loss_fn = torch_metrics.MaskedMSE()
    elif cfg.model.name in ['unkrig', 'unkrigv2', 'unkrigv3', 'unkrigv4']:
        loss_fn = [torch_metrics.MaskedMAE(), torch_metrics.MaskedMSE()]
    elif cfg.model.name == 'ignnk':
        loss_fn = torch_metrics.MaskedMSE()
    elif cfg.model.name in ['kits', 'unkrigv5', 'lsjstn', 'dida', 'caustg']:
        loss_fn = torch_metrics.MaskedMAE()
    else:
        raise f'{cfg.model.name} not implemented'

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mre': torch_metrics.MaskedMRE(),
        'mape': torch_metrics.MaskedMAPE()
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup imputer
    if cfg.model.name =='kits':
        imputer = GCNCycVirtualFiller(model_class=model_cls,
                                    model_kwargs=model_kwargs,
                                    optim_class=getattr(torch.optim, cfg.optimizer.name),
                                    optim_kwargs=dict(cfg.optimizer.hparams),
                                    loss_fn=loss_fn,
                                    scaled_target=cfg.scale_target,
                                    whiten_prob=cfg.whiten_prob,
                                    pred_loss_weight=cfg.prediction_loss_weight,
                                    warm_up=cfg.warm_up_steps,
                                    metrics=log_metrics,
                                    scheduler_class=scheduler_class,
                                    scheduler_kwargs=scheduler_kwargs,
                                    gradient_clip_val=cfg.grad_clip_val,
                                    gradient_clip_algorithm=cfg.grad_clip_alg,
                                    known_nodes = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                                    **cfg.model.technique)
    elif cfg.model.name == "unkrig":
        imputer = UnnamedKrigFiller(model_class=model_cls,
                                    model_kwargs=model_kwargs,
                                    optim_class=[getattr(torch.optim, cfg.optimizer.name_1), 
                                                 getattr(torch.optim, cfg.optimizer.name_2)],
                                    optim_kwargs=[dict(cfg.optimizer.hparams_1),
                                                  dict(cfg.optimizer.hparams_2)],
                                    loss_fn=loss_fn,
                                    scaled_target=cfg.scale_target,
                                    whiten_prob=cfg.whiten_prob,
                                    pred_loss_weight=cfg.prediction_loss_weight,
                                    warm_up=cfg.warm_up_steps,
                                    metrics=log_metrics,
                                    scheduler_class=scheduler_class,
                                    scheduler_kwargs=scheduler_kwargs,
                                    gradient_clip_val=cfg.grad_clip_val,
                                    gradient_clip_algorithm=cfg.grad_clip_alg,
                                    known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                                    **cfg.model.regs)
    elif cfg.model.name == "unkrigv2" or cfg.model.name == "unkrigv3":
        imputer = UnnamedKrigFillerV2(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=[getattr(torch.optim, cfg.optimizer.name_1), 
                                         getattr(torch.optim, cfg.optimizer.name_2)],
                            optim_kwargs=[dict(cfg.optimizer.hparams_1),
                                          dict(cfg.optimizer.hparams_2)],
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name == "unkrigv4":
        imputer = UnnamedKrigFillerV4(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=[getattr(torch.optim, cfg.optimizer.name_1), 
                                         getattr(torch.optim, cfg.optimizer.name_2)],
                            optim_kwargs=[dict(cfg.optimizer.hparams_1),
                                          dict(cfg.optimizer.hparams_2)],
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name == "unkrigv5":
        imputer = UnnamedKrigFillerV5(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name == "dida":
        imputer = DidaFiller(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            adj=adj,
                            horizon=cfg.window,
                            **cfg.model.regs)
    elif cfg.model.name == "caustg":
        imputer = CauSTGFiller(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            adj=adj,
                            device=cfg.device,
                            **cfg.model.regs)
    elif cfg.model.name == "ignnk":
        imputer = IGNNKFiller(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            adj=adj,
                            n_o_n_m=cfg.model.n_o_n_m)
    elif cfg.model.name == "lsjstn":
        imputer = LSJSTNFiller(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            whiten_prob=cfg.whiten_prob,
                            pred_loss_weight=cfg.prediction_loss_weight,
                            warm_up=cfg.warm_up_steps,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            adj=adj)
    else:
        imputer = Imputer(model_class=model_cls,
                        model_kwargs=model_kwargs,
                        optim_class=getattr(torch.optim, cfg.optimizer.name),
                        optim_kwargs=dict(cfg.optimizer.hparams),
                        loss_fn=loss_fn,
                        metrics=log_metrics,
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        scale_target=cfg.scale_target,
                        whiten_prob=cfg.whiten_prob,
                        prediction_loss_weight=cfg.prediction_loss_weight,
                        impute_only_missing=cfg.impute_only_missing,
                        warm_up_steps=cfg.warm_up_steps)

    ########################################
    # logging options                      #
    ########################################

    if 'wandb' in cfg:
        exp_logger = WandbLogger(name=cfg.run.name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.wandb.offline,
                                 project=cfg.wandb.project)
    elif cfg.logger == 'tensorboard':
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir,
                                       name='tensorboard')
    else: 
        exp_logger = None

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min',
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    if cfg.model.name in MODELS:
        trainer = Trainer(
            max_epochs=cfg.epochs,
            default_root_dir=cfg.run.dir,
            logger=exp_logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=cfg.device,
            callbacks=[early_stop_callback, checkpoint_callback],
            detect_anomaly=False)
    else:
        trainer = Trainer(
            max_epochs=cfg.epochs,
            default_root_dir=cfg.run.dir,
            logger=exp_logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=cfg.device,
            gradient_clip_val=cfg.grad_clip_val,
            gradient_clip_algorithm=cfg.grad_clip_alg,
            callbacks=[early_stop_callback, checkpoint_callback])


    trainer.fit(imputer, datamodule=dm, ckpt_path=cfg.call_path)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    
    if cfg.eval_setting == 'train_wise':
        res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
                test_mre=numpy_metrics.mre(y_hat, y_true, mask),
                test_mape=numpy_metrics.mape(y_hat, y_true, mask),
                test_mse=numpy_metrics.mse(y_hat, y_true, mask),
                test_rmse=numpy_metrics.rmse(y_hat, y_true, mask))
    elif cfg.eval_setting == 'test_wise':
        res = test_wise_eval(y_hat, y_true, mask, 
                             known_nodes=[i for i in range(adj.shape[0]) if i not in masked_sensors],
                             adj=adj,
                             mode='test')

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    
    if cfg.eval_setting == 'train_wise':
        res.update(
            dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
                val_mre=numpy_metrics.mre(y_hat, y_true, mask),
                val_mape=numpy_metrics.mape(y_hat, y_true, mask),
                val_mse=numpy_metrics.mse(y_hat, y_true, mask),
                val_rmse=numpy_metrics.rmse(y_hat, y_true, mask)))
    elif cfg.eval_setting == 'test_wise':
        res.update(test_wise_eval(y_hat, y_true, mask, 
                    known_nodes=[i for i in range(adj.shape[0]) if i not in masked_sensors],
                    adj=adj, mode='val'))
    
    res.update(
        dict(model=cfg.model.name,
             db=cfg.dataset.name,
             seed=cfg.seed,
             mode=cfg.dataset.mode,
             spatial=cfg.dataset.spatial_shift,
             eval_setting=cfg.eval_setting)
    )

    return res

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)