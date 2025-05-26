pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install pandas==1.3.5
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install matplotlib scikit-learn ipykernel tensorboard haversine omegaconf tables seaborn torch-geometric-temporal networkx
pip install git+https://github.com/TorchSpatiotemporal/tsl.git 
pip install hydra-core POT numba