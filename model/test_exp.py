import argparse
import os
import random

import numpy as np
import torch
from geode import Geode
from geode_filler import GeodeFiller
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PeMS04, PeMS07, PvUS
from tsl.metrics import numpy as numpy_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy
from utils import add_missing_sensors, test_wise_eval, LargeST


def get_dataset(dataset_name: str, p_fault=0., p_noise=0., masked_s=None, connectivity=None, mode='road',
                spatial_shift=False, order=0, node_features='CC'):
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

def load_model_and_infer(config_path: str, checkpoint_path: str):
    torch.set_float32_matmul_precision('high')   
    # Load configuration
    cfg = OmegaConf.load(config_path)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Load dataset
    dataset, masked_sensors = get_dataset(cfg.dataset.name,
                            p_fault=cfg.dataset.get('p_fault'),
                            p_noise=cfg.dataset.get('p_noise'),
                            masked_s=cfg.dataset.get('masked_sensors'),
                            connectivity=cfg.dataset.get('connectivity'),
                            spatial_shift=cfg.dataset.get('spatial_shift'),
                            order=cfg.dataset.get('order'),
                            node_features=cfg.dataset.get('node_features'))
    print(f'Masked sensors: {masked_sensors}')
    adj = dataset.get_connectivity(**cfg.dataset.connectivity, layout='dense')
    
    torch_dataset = ImputationDataset(
        target=dataset.dataframe(),
        mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        transform=MaskInput(),
        connectivity=adj,
        window=cfg.window,
        stride=cfg.stride
    )
    
    scalers = {'target': StandardScaler(axis=(0, 1))}
    val_len = cfg.dataset.splitting.get('val_len')
    test_len = cfg.dataset.splitting.get('test_len')
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(val_len=val_len, test_len=test_len),
        batch_size=cfg.batch_size,
        workers=cfg.workers)
    dm.setup(stage='test')
    
    # Load model
    model_cls = Geode
    model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.device,
        gradient_clip_val=cfg.get('grad_clip_val', None),
        gradient_clip_algorithm=cfg.get('grad_clip_alg', None),
        logger=False)
    
    imputer = GeodeFiller.load_from_checkpoint(checkpoint_path, model_class=model_cls,
                          model_kwargs=model_kwargs,
                          gradient_clip_val=cfg.grad_clip_val,
                          gradient_clip_algorithm=cfg.grad_clip_alg,
                          known_set = [i for i in range(adj.shape[0]) if i not in masked_sensors],
                          **cfg.model.regs)
    imputer.eval()
    
    # Run inference
    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)

    y_hat, y_true, mask = (output['y_hat'], output['y'], output.get('eval_mask', None))
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
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model inference with config and checkpoint.')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint.')
    args = parser.parse_args()

    # Assuming this function is defined elsewhere
    results = load_model_and_infer(args.config, args.checkpoint)
    print("Inference results:", results)