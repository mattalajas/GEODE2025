import csv
import os

from tsl import logger
from tsl.experiment import Experiment
from Grin import run_imputation

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)

    mode = res.pop('mode')
    shift = res.pop('spatial')
    eval_setting = res.pop('eval_setting')
    node_features = res.pop('node_f')
    if shift:
        csv_path = f"/data/mala711/Thesis/GNNthesis/res/{res['model']}-{mode}-spatial-{node_features}.csv"
    else:
        if eval_setting == 'train_wise':
            csv_path = f"/data/mala711/Thesis/GNNthesis/res/{res['model']}-{mode}.csv"
        else:
            csv_path = f"/data/mala711/Thesis/GNNthesis/res/{res['model']}-{mode}-testwise.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(res)