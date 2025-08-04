# GEODE: Graph Feature Invariance for Inductive Spatiotemporal Learning
This repository contains the codebase for GEODE.

## Requirements
Main package requirements:
- ```Python == 3.10```
- ```PyTorch == 2.1.0```
- ```PyTorch Geometric == 2.6.1```
  - ```pyg_lib == 0.3.1```
  - ```torch_scatter == 2.1.2```
  - ```torch_sparse == 0.6.18```
  - ```torch_cluster == 1.6.3```
  - ```torch_spline_conv == 1.2.2```
- ```CUDA == 12.1```
- Torch Spatiotemporal Library (TSL): [https://github.com/TorchSpatiotemporal/tsl]

Install other packages through a conda environment using the following command from the root directory of the repository.

```
conda env create -f environment.yml
```

## Dataset
We use the following datasets:
- AQI and AQI-SM
- METR-LA
- NREL-AL and NREL-MD
- PEMS07 and PEMS04
- SD

The datasets we used are available under the TSL codebase, and is downloaded within the training pipeline.

## Usage

All hyperparameter and dataset configurations are located in ```\model\config```. Specifically:
- ```\model\config\default.yaml``` contains general training configurations and hyperparameters.
- ```\model\config\dataset\*.yaml``` contains specific dataset configurations.
- ```\model\config\model\geode.yaml``` contains specific geode hyperparameters.

To train GEODE on all datasets, run the following from the root directory.
```
bash run_exp-geode-trainwise-rnd.bash   #For trainwise selection - RND
bash run_exp-geode-trainwise-rnd.bash   #For trainwise selection - CC
bash run_exp-geode-testwise.bash        #For testwise selection
```

To train GEODE on a single run, run the following from the root directory.
```
python3 model/mult_experiments.py 'dataset=aqism' 'model.hparams.k=2' 'model.hparams.psd_layers=2' 'model.hparams.gcn_layers=1' 'optimizer.hparams.lr=0.0001' 'model.regs.y2=0.5' --config-name=default
```

Saved models are in ```/logs```, while csv results are in ```/res```.

To test a saved GEODE model, run the following from the root directory.
```
python3 test_exp.py --config=<config_path.yaml> --checkpoint=<checkpoint_path.ckpt>
```

Trained model configs and checkpoint paths are located in ```/logs```.
Training results are located in ```/res```.
