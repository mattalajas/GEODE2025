import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import numpy as np

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PeMS07, PvUS, LargeST, PeMS04
from tsl.experiment import Experiment
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

from utils import add_missing_sensors

from geode import Geode
from geode_filler import GeodeFiller
from utils import test_wise_eval

def get_dataset(dataset_name: str, p_fault=0., p_noise=0., masked_s=None, connectivity=None, mode='road',
                spatial_shift=False, order=0, node_features='c_centrality'):
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
                            masked_s=cfg.dataset.get('masked_sensors'),
                            connectivity=cfg.dataset.get('connectivity'),
                            mode=cfg.dataset.get('mode'),
                            spatial_shift=cfg.dataset.get('spatial_shift'),
                            order=cfg.dataset.get('order'),
                            node_features=cfg.dataset.get('node_features'))

    print(f'Masked sensors: {masked_sensors}')

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    # instantiate dataset
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=cfg.window,
                                      stride=cfg.stride)

    scalers = {'target': StandardScaler(axis=(0, 1))}
    val_len = cfg.dataset.splitting.get('val_len')
    test_len = cfg.dataset.splitting.get('test_len')
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(val_len=val_len, test_len=test_len),
        batch_size=cfg.batch_size,
        workers=cfg.workers)
    dm.setup(stage='fit')

    ########################################
    # imputer                              #
    ########################################

    model_cls = Geode

    model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
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
    
    imputer = GeodeFiller(model_class=model_cls,
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
                test_mape=numpy_metrics.mape(y_hat, y_true, mask),
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
                val_mape=numpy_metrics.mape(y_hat, y_true, mask),
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
             mode=cfg.dataset.mode,
             spatial=cfg.dataset.spatial_shift,
             eval_setting=cfg.eval_setting,
             node_f=cfg.dataset.node_features)
    )
    return res

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)