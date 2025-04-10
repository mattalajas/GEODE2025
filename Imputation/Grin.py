import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import numpy as np

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.engines import Imputer
from tsl.experiment import Experiment
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import (BiRNNImputerModel, GRINModel, RNNImputerModel,
                           SPINHierarchicalModel, SPINModel)
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

from my_datasets import AirQualitySmaller, AirQualityAuckland, AirQualityKrig, add_missing_sensors
from KITS import KITS
from KITS_filler import GCNCycVirtualFiller
from unnamed_filler import UnnamedKrigFiller
from unnamedKrig import UnnamedKrigModel


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
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name: str, p_fault=0., p_noise=0., t_range = ['2022-04-01', '2022-12-01'],
                masked_s=None, agg_func = 'mean', test_month=[5], location='Auckland'):
    if dataset_name == 'air':
        return AirQualityKrig(impute_nans=True, small=True, masked_sensors=masked_s, p=p_noise)
    if dataset_name == 'air_smaller':
        return AirQualitySmaller('../../AirData/AQI/Stations', impute_nans=True, masked_sensors=masked_s)
    if dataset_name == 'air_auckland' or dataset_name == 'air_invercargill1' or dataset_name == 'air_invercargill2':
        return AirQualityAuckland('../../AirData/Niwa', t_range=t_range, masked_sensors=masked_s, 
                                  agg_func=agg_func, test_months=test_month,
                                  location=location, p=p_noise)
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    if dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    if dataset_name == 'metrla':
        return add_missing_sensors(MetrLA(freq='5T'),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=9101112, 
                                  masked_sensors=masked_s)
    if dataset_name == 'bay':
        return add_missing_sensors(PemsBay(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=56789,
                                  masked_sensors=masked_s)
    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def run_imputation(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    torch.set_float32_matmul_precision('high')

    dataset = get_dataset(cfg.dataset.name,
                        p_fault=cfg.dataset.get('p_fault'),
                        p_noise=cfg.dataset.get('p_noise'),
                        t_range=cfg.dataset.get('t_range'),
                        masked_s=cfg.dataset.get('masked_sensors'),
                        agg_func=cfg.dataset.get('agg_func'),
                        test_month=cfg.dataset.get('test_month'),
                        location=cfg.dataset.get('location'))

    # encode time of the day and use it as exogenous variable
    # covariates = {'u': dataset.datetime_encoded('day').values}

    # get adjacency matrix
    if cfg.model.name == 'kits' or cfg.model.name == 'unkrig':
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

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers)
    dm.setup(stage='fit')

    if cfg.get('in_sample', False):
        dm.trainset = list(range(len(torch_dataset)))

    ########################################
    # imputer                              #
    ########################################

    model_cls = get_model_class(cfg.model.name)
    
    if cfg.model.name == 'kits':
        model_kwargs = dict(adj=adj, d_in=dm.n_channels, n_nodes=dm.n_nodes, args=cfg.model)
    elif cfg.model.name == 'unkrig':
        model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels)
    else:
        model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                            input_size=torch_dataset.n_channels)
                            # ,exog_size=torch_dataset.input_map.u.shape[-1])

    model_cls.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    loss_fn = torch_metrics.MaskedMAE()

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
                                    known_nodes = [i for i in range(adj.shape[0]) if i not in cfg.dataset.get('masked_sensors')],
                                    **cfg.model.technique)
    elif cfg.model.name == "unkrig":
        imputer = UnnamedKrigFiller(model_class=model_cls,
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
                                    **cfg.model.regs)
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

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    if cfg.model.name =='kits' or cfg.model.name =='unkrig':
        trainer = Trainer(
            max_epochs=cfg.epochs,
            default_root_dir=cfg.run.dir,
            logger=exp_logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=cfg.device,
            callbacks=[early_stop_callback, checkpoint_callback],
            detect_anomaly=True)
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


    trainer.fit(imputer, datamodule=dm)

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
    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
               test_mre=numpy_metrics.mre(y_hat, y_true, mask),
               test_mape=numpy_metrics.mape(y_hat, y_true, mask),
               test_mse=numpy_metrics.mse(y_hat, y_true, mask),
               test_rmse=numpy_metrics.rmse(y_hat, y_true, mask))

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
             val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
             val_mape=numpy_metrics.mape(y_hat, y_true, mask),
             val_mse=numpy_metrics.mse(y_hat, y_true, mask)))

    return res


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)