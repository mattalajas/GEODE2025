import random

import numpy as np
import torch
from geode import Geode
from geode_filler import GeodeFiller
from geodeCross import GeodeCross
from geodeCrossv2 import GeodeCrossV2
from geodeCrossv3 import GeodeCrossV3
from geodeCrossv4 import GeodeCrossV4
from geodeCrossv5 import GeodeCrossV5
from geodeCrossv6 import GeodeCrossV6
from geodeCrossv7 import GeodeCrossV7
from geodeCrossv8 import GeodeCrossV8
from geodeCrossv9 import GeodeCrossV9
from geodeCrossv10 import GeodeCrossV10
from geodeCrossv11 import GeodeCrossV11
from geodeCrossv12 import GeodeCrossV12
from geodeCrossv13 import GeodeCrossV13
from geodeCrossv14 import GeodeCrossV14
from geodeCrossc1 import GeodeCrossC1
from geodeCrossc2 import GeodeCrossC2
from geodeCross_filler import GeodeCrossFiller
from geodeCross_fillerV4 import GeodeCrossFillerV4
from geodeCross_fillerV5 import GeodeCrossFillerV5
from geodeCross_fillerV6 import GeodeCrossFillerV6
from geodeCross_fillerV7 import GeodeCrossFillerV7
from geodeCross_fillerV8 import GeodeCrossFillerV8
from geodeCross_fillerC1 import GeodeCrossFillerC1
from geodeNAall import GeodeNAall
from KITS import KITS
from KITS_filler import GCNCycVirtualFiller
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tsl import logger
from tsl.data import ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PeMS04, PeMS07, PvUS, GPVARDatasetAZ
from tsl.experiment import Experiment
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy
from utils import (AirCross, AirQualityAuckland, CrossSpatioTemporalDataset,
                   LargeST, StandardScalerSplit, SpatioTemporalDataModule,
                   CrossGPVARDataset, add_missing_sensors, add_missing_sensors_cross, test_wise_eval)


def get_model_class(model_str):
    if model_str == 'geode':
        model = Geode
    elif model_str == 'geodeNAall':
        model = GeodeNAall
    elif model_str == 'geodeCross':
        model = GeodeCross
    elif model_str == 'geodeCrossV2':
        model = GeodeCrossV2
    elif model_str == 'geodeCrossV3':
        model = GeodeCrossV3
    elif model_str == 'geodeCrossV4':
        model = GeodeCrossV4
    elif model_str == 'geodeCrossV5':
        model = GeodeCrossV5
    elif model_str == 'geodeCrossV6':
        model = GeodeCrossV6
    elif model_str == 'geodeCrossV7':
        model = GeodeCrossV7
    elif model_str == 'geodeCrossV8':
        model = GeodeCrossV8
    elif model_str == 'geodeCrossV9':
        model = GeodeCrossV9
    elif model_str == 'geodeCrossV10':
        model = GeodeCrossV10
    elif model_str == 'geodeCrossV11':
        model = GeodeCrossV11
    elif model_str == 'geodeCrossV12':
        model = GeodeCrossV12
    elif model_str == 'geodeCrossV13':
        model = GeodeCrossV13
    elif model_str == 'geodeCrossV14':
        model = GeodeCrossV14
    elif model_str == 'geodeCrossC1':
        model = GeodeCrossC1
    elif model_str == 'geodeCrossC2':
        model = GeodeCrossC2
    elif model_str == 'kits':
        model = KITS
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model

def get_dataset(dataset_name: str, p_fault=0., p_noise=0., masked_s=None, connectivity=None, 
                spatial_shift=False, order=0, node_features='CC', t_range = ['2022-04-01', '2022-12-01'], 
                agg_func = 'mean', test_months=[5], location='Auckland', 
                years = [], include_exog=False, exog='traffic', synth_params=None):
    
    if dataset_name == 'air_auckland' or dataset_name == 'air_invercargill1' or dataset_name == 'air_invercargill2':
        air_data = AirQualityAuckland('data', t_range=t_range, masked_sensors=masked_s, 
                                  agg_func=agg_func, test_months=test_months,
                                  location=location, p=p_noise)
        
        return add_missing_sensors(air_data,
                            p_fault=p_fault,
                            p_noise=p_noise,
                            min_seq=12,
                            max_seq=12 * 4, 
                            masked_sensors=masked_s,
                            connect=connectivity,
                            spatial_shift=spatial_shift,
                            order=order,
                            node_features=node_features)
    
    if dataset_name == 'aqi':
        return add_missing_sensors(AirQuality(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4, 
                                  masked_sensors=masked_s,
                                  connect=connectivity,
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
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'sd':
        return add_missing_sensors(LargeST(subset='SD', year=[2018, 2019, 2020]),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'synthetic':
        return add_missing_sensors(GPVARDatasetAZ(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'syntheticCross':
        return add_missing_sensors_cross(CrossGPVARDataset(include_exog=include_exog, **synth_params),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'aircross':
        return add_missing_sensors_cross(AirCross(root='data/AirCrossSF', test_months=test_months,
                                                  years=years, include_exog=include_exog, exog=exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)
    if dataset_name == 'aircross_la':
        return add_missing_sensors_cross(AirCross(root='data/AirCrossLA', test_months=test_months,
                                                  years=years, include_exog=include_exog, exog=exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)
    if dataset_name == 'aircross_sp':
        return add_missing_sensors_cross(AirCross(root='data/AirCrossSpain', test_months=test_months,
                                                  years=years, include_exog=include_exog, exog=exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)
    if dataset_name == 'aircross_auck':
        return add_missing_sensors_cross(AirCross(root='data/AucklandCross', test_months=test_months,
                                                  years=years, include_exog=include_exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)

    raise ValueError(f"Dataset {dataset_name} not available in this setting.")

def run_imputation(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    torch.set_float32_matmul_precision('high')
    # Load configuration
    
    seed_everything(cfg.seed, workers=True)

    assert cfg.eval_setting in ['train_wise', 'test_wise']

    dataset, masked_sensors = get_dataset(cfg.dataset.name,
                            p_fault=cfg.dataset.get('p_fault'),
                            p_noise=cfg.dataset.get('p_noise'),
                            masked_s=cfg.dataset.get('masked_sensors'),
                            connectivity=cfg.dataset.get('connectivity'),
                            spatial_shift=cfg.dataset.get('spatial_shift'),
                            order=cfg.dataset.get('order'),
                            node_features=cfg.dataset.get('node_features'),
                            t_range=cfg.dataset.get('t_range'),
                            agg_func=cfg.dataset.get('agg_func'),
                            location=cfg.dataset.get('location'),
                            test_months=cfg.dataset.get('test_months', (3, 6, 9, 12)),
                            years=cfg.dataset.get('years', []),
                            include_exog=cfg.dataset.get('include_exog', False),
                            exog=cfg.dataset.get('exog', 'traffic'),
                            synth_params=cfg.dataset.get('synth_params', None))

    print(f'Masked sensors: {masked_sensors}')

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity, layout='dense')

    # instantiate dataset
    if cfg.dataset.get('include_exog', False):
        covariates = {'modality': dataset.modality}
        torch_dataset = CrossSpatioTemporalDataset(target=dataset.dataframe(),
                                                    mask=dataset.training_mask,
                                                    eval_mask=dataset.eval_mask,
                                                    covariates=covariates,
                                                    transform=MaskInput(),
                                                    connectivity=adj,
                                                    window=cfg.window,
                                                    stride=cfg.stride)
    else:
        torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                        mask=dataset.training_mask,
                                        eval_mask=dataset.eval_mask,
                                        transform=MaskInput(),
                                        connectivity=adj,
                                        window=cfg.window,
                                        stride=cfg.stride)

    if cfg.scaler == 'StandardScaler':
        scalers = {'target': StandardScaler(axis=(0, 1))}
    elif cfg.scaler == 'StandardScalerSplit':
        scalers = {'target': StandardScalerSplit(axis=(0, 1), split=dataset.air_max_nodes)}
    else:
        raise ValueError(f"Scaler {cfg.scaler} not available in this setting.")

    val_len = cfg.dataset.splitting.get('val_len')
    test_len = cfg.dataset.splitting.get('test_len')

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(val_len=val_len, test_len=test_len),
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        generator=g)
    dm.setup(stage='fit')

    ########################################
    # imputer                              #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    if cfg.model.name == 'kits':
        model_kwargs = dict(adj=adj, d_in=dm.n_channels, n_nodes=dm.n_nodes, args=cfg.model)
    elif cfg.model.name =='geode' or cfg.model.name == 'geodeNAall' \
        or cfg.model.name == 'geodeCross' or cfg.model.name == 'geodeCrossV2' \
        or cfg.model.name == 'geodeCrossV3' or cfg.model.name == 'geodeCrossV4' \
        or cfg.model.name == 'geodeCrossV5' or cfg.model.name == 'geodeCrossV6' \
        or cfg.model.name == 'geodeCrossV7' or cfg.model.name == 'geodeCrossV8' \
        or cfg.model.name == 'geodeCrossV9' or cfg.model.name == 'geodeCrossV10' \
        or cfg.model.name == 'geodeCrossV11' or cfg.model.name == 'geodeCrossV12' \
        or cfg.model.name == 'geodeCrossC1' or cfg.model.name == 'geodeCrossV13' \
        or cfg.model.name == 'geodeCrossV14':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    elif cfg.model.name == 'geodeCrossC2':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window, threshold=cfg.dataset.connectivity.threshold)

    model_cls.filter_model_args_(model_kwargs)
    loss_fn = torch_metrics.MaskedMAE()

    model_kwargs.update(cfg.model.hparams)

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mre': torch_metrics.MaskedMRE()
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    if cfg.model.name == 'kits':
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
    elif cfg.model.name =='geode' or cfg.model.name =='geodeNAall':
        imputer = GeodeFiller(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCross' or cfg.model.name =='geodeCrossV2' or cfg.model.name =='geodeCrossV3':
        imputer = GeodeCrossFiller(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCrossV4' or cfg.model.name == 'geodeCrossV5' or \
        cfg.model.name == 'geodeCrossV6' or cfg.model.name == 'geodeCrossV7':
        imputer = GeodeCrossFillerV4(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCrossV8':
        imputer = GeodeCrossFillerV5(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCrossV9' or cfg.model.name =='geodeCrossV10':
        imputer = GeodeCrossFillerV6(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCrossV11':
        imputer = GeodeCrossFillerV7(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCrossV12' or cfg.model.name =='geodeCrossV13' \
        or cfg.model.name =='geodeCrossV14':
        imputer = GeodeCrossFillerV8(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    elif cfg.model.name =='geodeCrossC1' or cfg.model.name =='geodeCrossC2':
        imputer = GeodeCrossFillerC1(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=getattr(torch.optim, cfg.optimizer.name),
                            optim_kwargs=dict(cfg.optimizer.hparams),
                            loss_fn=loss_fn,
                            scaled_target=cfg.scale_target,
                            metrics=log_metrics,
                            scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs,
                            gradient_clip_val=cfg.grad_clip_val,
                            gradient_clip_algorithm=cfg.grad_clip_alg,
                            known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                            **cfg.model.regs)
    else:
        raise NotImplementedError(f'Model "{cfg.model.name}" not available.')

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
    trainer = Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.device,
        callbacks=[early_stop_callback, checkpoint_callback],
        detect_anomaly=False)
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
                test_mse=numpy_metrics.mse(y_hat, y_true, mask),
                test_rmse=numpy_metrics.rmse(y_hat, y_true, mask))
    elif cfg.eval_setting == 'test_wise':
        res = test_wise_eval(y_hat, y_true, mask, 
                             known_nodes=[i for i in range(adj.shape[0]) if i not in masked_sensors],
                             adj=adj,
                             mode='test',
                             num_groups=cfg.num_groups)

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    
    if cfg.eval_setting == 'train_wise':
        res.update(
            dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
                val_mre=numpy_metrics.mre(y_hat, y_true, mask),
                val_mse=numpy_metrics.mse(y_hat, y_true, mask),
                val_rmse=numpy_metrics.rmse(y_hat, y_true, mask)))
    elif cfg.eval_setting == 'test_wise':
        res.update(test_wise_eval(y_hat, y_true, mask, 
                    known_nodes=[i for i in range(adj.shape[0]) if i not in masked_sensors],
                    adj=adj, mode='val', num_groups=cfg.num_groups))
    
    res.update(
        dict(model=cfg.model.name,
             db=cfg.dataset.name,
             seed=cfg.seed,
             spatial=cfg.dataset.spatial_shift,
             eval_setting=cfg.eval_setting,
             node_f=cfg.dataset.node_features)
    )
    return res

if __name__ == '__main__':
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)