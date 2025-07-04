import inspect
import warnings
from copy import deepcopy
from einops import rearrange

import numpy as np
import pytorch_lightning as pl
import torch
from active_learning import GraphStorage
from einops import rearrange
from pytorch_lightning.utilities import move_data_to_device
from torch_geometric.utils import dense_to_sparse
from torchmetrics import MetricCollection
from tsl import logger
from tsl.metrics.torch import MaskedMetric
from tsl.metrics.torch.functional import mre

warnings.filterwarnings("ignore")

NCL = 10


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]

class Filler(pl.LightningModule):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 adj=None,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):
        """
        PL module to implement hole fillers.

        :param model_class: Class of pytorch nn.Module implementing the imputer.
        :param model_kwargs: Model's keyword arguments.
        :param optim_class: Optimizer class.
        :param optim_kwargs: Optimizer's keyword arguments.
        :param loss_fn: Loss function used for training.
        :param scaled_target: Whether to scale target before computing loss using batch processing information.
        :param whiten_prob: Probability of removing a value and using it as ground truth for imputation.
        :param metrics: Dictionary of type {'metric1_name':metric1_fn, 'metric2_name':metric2_fn ...}.
        :param scheduler_class: Scheduler class.
        :param scheduler_kwargs: Scheduler's keyword arguments.
        """
        super(Filler, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn'], logger=False)
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.automatic_optimization = False
        self.adj = adj

        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs = scheduler_kwargs

        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        self.scaled_target = scaled_target

        # during training whiten ground-truth values with this probability
        assert 0. <= whiten_prob <= 1.
        self.keep_prob = 1. - whiten_prob

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)
        # instantiate model
        self.model = self.model_cls(**self.model_kwargs)

    def reset_model(self):
        self.model = self.model_cls(**self.model_kwargs)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def collate_prediction_outputs(self, outputs):
        """
        Collate the outputs of the :meth:`predict_step` method.

        Args:
            outputs: Collated outputs of the :meth:`predict_step` method.

        Returns:
            The collated outputs.
        """
        # iterate over results
        processed_res = dict()
        keys = set()
        # iterate over outputs for each batch
        for res in outputs:
            for k, v in res.items():
                if k in keys:
                    processed_res[k].append(v)
                else:
                    processed_res[k] = [v]
                keys.add(k)
        # concatenate results
        for k, v in processed_res.items():
            processed_res[k] = torch.cat(v, 0)
        return processed_res
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # batch_data["training"] = False

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        x = batch_data.pop("x")
        mask = batch_data.pop("mask")

        adj = torch.tensor(self.adj).to(device=x.device)
        x = rearrange(x, 'b t n d -> t b n d')
        x = list(x)

        edge_index, edge_weights = dense_to_sparse(adj)

        batch_data["x_list"] = x  # t s n2 d
        batch_data["edge_index"] = edge_index
        batch_data["edge_weights"] = edge_weights

        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        
        output = dict(y=y,
                      y_hat=imputation,
                      eval_mask=eval_mask)
        return output

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric, compute_on_step=on_step, metric_kwargs=metric_kwargs)
        return deepcopy(metric)

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_mean/{name}', param.grad.mean(), on_step=True)
                self.log(f'grad_max/{name}', param.grad.max(), on_step=True)
                self.log(f'grad_min/{name}', param.grad.min(), on_step=True)      

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='train_')
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='val_', compute_groups=False)
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='test_')

    def _preprocess(self, data, batch_preprocessing):
        """
        Perform preprocessing of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to preprocess
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: preprocessed data
        """
        for key, trans in batch_preprocessing.items():
            if key in data:
                data[key] = trans.transform(data[key])
        return data

    def _postprocess(self, data, batch_preprocessing):
        """
        Perform postprocessing (inverse transform) of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to trasform
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: inverse transformed data
        """
        trans = batch_preprocessing.get('y')
        if trans is not None:
            data = trans.inverse_transform(data)
        return data

    def predict_batch(self, batch, preprocess=False, postprocess=True, return_target=False):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :return: (y_true), y_hat, (mask)
        """
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop('x')
            x = [self._preprocess(x[i], batch_preprocessing) for i in range(len(x))]
            y_hat = self.forward(x, **batch_data)
        else:
            y_hat = self.forward(**batch_data)
        # Rescale outputs
        if postprocess:
            y_hat = self._postprocess(y_hat, batch_preprocessing)
        if return_target:
            y = batch_data.get('y')
            mask = batch_data.get('mask', None)
            return y, y_hat, mask
        return y_hat

    def predict_loader(self, loader, preprocess=False, postprocess=True, return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, imputations, masks = [], [], []
        for batch in loader:
            batch = move_data_to_device(batch, self.device)
            batch_data, batch_preprocessing = self._unpack_batch(batch)
            # Extract mask and target
            eval_mask = batch_data.pop('eval_mask', None)
            y = batch_data.pop('y')

            x = batch_data.pop("x")
            mask = batch_data.pop("mask")

            adj = torch.tensor(self.adj).to(device=x.device)
            x = rearrange(x, 'b t n d -> t b n d')
            x = list(x)

            edge_index, edge_weights = dense_to_sparse(adj)

            batch_data["x_list"] = x  # t s n2 d
            batch_data["edge_index"] = edge_index
            batch_data["edge_weights"] = edge_weights

            y_hat = self.predict_batch(batch, preprocess=preprocess, postprocess=postprocess)

            targets.append(y)
            imputations.append(y_hat)
            masks.append(eval_mask)

        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.

        :param batch: the batch
        :return: batch_data, batch_preprocessing
        """
        batch_preprocessing = batch.get('transform')
        return batch, batch_preprocessing

    # def training_step(self, batch, batch_idx):
    #     # Unpack batch
    #     batch_data, batch_preprocessing = self._unpack_batch(batch)

    #     # Extract mask and target
    #     mask = batch_data['mask'].clone().detach()
    #     batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
    #     eval_mask = batch_data.pop('eval_mask')
    #     eval_mask = (mask | eval_mask) - batch_data['mask']

    #     y = batch_data.pop('y')

    #     # Compute predictions and compute loss
    #     imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

    #     if self.scaled_target:
    #         target = self._preprocess(y, batch_preprocessing)
    #     else:
    #         target = y
    #         imputation = self._postprocess(imputation, batch_preprocessing)

    #     loss = self.loss_fn(imputation, target, mask)

    #     # Logging
    #     if self.scaled_target:
    #         imputation = self._postprocess(imputation, batch_preprocessing)
    #     self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
    #     self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
    #     self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     # Unpack batch
    #     batch_data, batch_preprocessing = self._unpack_batch(batch)

    #     # Extract mask and target
    #     eval_mask = batch_data.pop('eval_mask', None)
    #     y = batch_data.pop('y')

    #     # Compute predictions and compute loss
    #     imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

    #     if self.scaled_target:
    #         target = self._preprocess(y, batch_preprocessing)
    #     else:
    #         target = y
    #         imputation = self._postprocess(imputation, batch_preprocessing)

    #     val_loss = self.loss_fn(imputation, target, eval_mask)

    #     # Logging
    #     if self.scaled_target:
    #         imputation = self._postprocess(imputation, batch_preprocessing)
    #     self.val_metrics.update(imputation.detach(), y, eval_mask)
    #     self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
    #     self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
    #     return val_loss

    # def test_step(self, batch, batch_idx):
    #     # Unpack batch
    #     batch_data, batch_preprocessing = self._unpack_batch(batch)

    #     # Extract mask and target
    #     eval_mask = batch_data.pop('eval_mask', None)
    #     y = batch_data.pop('y')

    #     # Compute outputs and rescale
    #     imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
    #     test_loss = self.loss_fn(imputation, y, eval_mask)

    #     # Logging
    #     self.test_metrics.update(imputation.detach(), y, eval_mask)
    #     self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
    #     return test_loss

    def on_train_epoch_start(self) -> None:
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def configure_optimizers(self):
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg

class DYSATFiller(Filler):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn=None,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 known_set=None,
                 adj=None,
                 horizon=24):
        super(DYSATFiller, self).__init__(model_class=model_class,
                                                  model_kwargs=model_kwargs,
                                                  optim_class=optim_class,
                                                  optim_kwargs=optim_kwargs,
                                                  loss_fn=loss_fn,
                                                  scaled_target=scaled_target,
                                                  whiten_prob=whiten_prob,
                                                  metrics=metrics,
                                                  adj=adj,
                                                  scheduler_class=scheduler_class,
                                                  scheduler_kwargs=scheduler_kwargs)
        
        self.tradeoff = pred_loss_weight
        self.trimming = (warm_up, warm_up)

        self.known_set = None
        self.adj = adj
        self.horizon = horizon

        self.known_set = known_set

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq
    
    def load_model(self, filename: str):
        """Load model's weights from checkpoint at :attr:`filename`.

        Differently from
        :meth:`~pytorch_lightning.core.LightningModule.load_from_checkpoint`,
        this method allows to load the state_dict also for models instantiated
        outside the predictor, without checking that hyperparameters of the
        checkpoint's model are the same of the predictor's model.
        """
        storage = torch.load(filename, lambda storage, loc: storage, weights_only=False)
        # if predictor.model has been instantiated inside predictor
        if self.model_cls is not None:
            model_cls = storage['hyper_parameters']['model_class']
            model_kwargs = storage['hyper_parameters']['model_kwargs']
            # check model class and hyperparameters are the same
            assert model_cls == self.model_cls
            # if model_kwargs is not None:
            #     for k, v in model_kwargs.items():
            #         assert v == self.model_kwargs[k], f'{v}'
        else:
            logger.warning("Predictor with already instantiated model is "
                           f"loading a state_dict from {filename}. Cannot "
                           " check if model hyperparameters are the same.")
        self.load_state_dict(storage['state_dict'])
    
    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      logger=True,
                      prog_bar=True,
                      **kwargs)

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(name + '_loss',
                 loss.detach(),
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False,
                 **kwargs)

    # def on_train_batch_start(self, batch, batch_idx):
    #     if self.e_start:
    #         batch_data, batch_preprocessing = self._unpack_batch(batch)

    #         # To make the model inductive
    #         # => remove unobserved entries from input data and adjacency matrix
    #         # if self.known_set is None:
    #             # Get observed entries (nonzero masks across time)
    #         mask = batch_data["mask"]
    #         mask = rearrange(mask, "b s n 1 -> (b s) n")
    #         mask_sum = mask.sum(0)  # n
    #         known_set = torch.where(mask_sum > 0)[0]
    #         ratio = float(len(known_set) / mask_sum.shape[0])
    #         self.ratio = ratio / 2

    #         dynamic_ratio = self.ratio + 0.2 * np.random.random()
    #         val_len = int(dynamic_ratio*len(known_set))
    #         arrange = torch.randperm(len(known_set), device=mask.device)
    #         val_set = known_set[arrange[:val_len]].detach().cpu().numpy().tolist()
    #         train_set = known_set[arrange[val_len:]].detach().cpu().numpy().tolist()

    #         self.train_set = train_set
    #         self.val_set = val_set
    #         self.e_start = False

    def training_step(self, batch, batch_idx):
        # Unpack batch
        opt = self.optimizers()
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        mask = batch_data["mask"]
        mask = rearrange(mask, "b s n 1 -> (b s) n")
        mask_sum = mask.sum(0)  # n
        
        if self.known_set is None:
            # Get observed entries (nonzero masks across time)
            known_set = torch.where(mask_sum > 0)[0].detach().cpu().numpy().tolist()
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio
        else:
            known_set = self.known_set
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio

        # batch_data["known_set"] = known_set

        x = batch_data.pop("x")
        mask = batch_data.pop("mask")
        y = batch_data.pop("y")
        _ = batch_data.pop("eval_mask")  # drop this, we will re-create a new eval_mask (=mask during training)
        adj = torch.tensor(self.adj).to(device=x.device)

        sub_entry_num = 0

        # # Create randomised model here
        # if self.inductive:
        b, s, n, d = mask.size()
        cur_entry_num = n

        dynamic_ratio = self.ratio + 0.2 * np.random.random()  # ratio + 0.1
        aug_entry_num = max(int(cur_entry_num / dynamic_ratio), cur_entry_num + 1)
        sub_entry_num = aug_entry_num - cur_entry_num  # n2 - n1
        assert sub_entry_num > 0, "The augmented data should have more entries than original data."
        self.sub_entry_num = sub_entry_num

        arrange = torch.randperm(len(known_set))
        t_set = torch.tensor(known_set)

        masked_indx = t_set[arrange[-sub_entry_num:]].numpy().tolist()
        seened_indx = t_set[arrange[:-sub_entry_num]].numpy().tolist()
        full = seened_indx + masked_indx

        x[:, :, masked_indx, :] = 0
        x = x[:, :, full, :]  # b s n1 d, n1 = num of observed entries
        x = rearrange(x, 'b t n d -> t b n d')
        x = list(x)

        mask[:, :, seened_indx, :] = 0
        mask = mask[:, :, full, :]  # b s n1 d
        y = y[:, :, full, :]  # b s n1 d

        adj = adj[:, full]
        adj = adj[full, :]

        edge_index, edge_weights = dense_to_sparse(adj)

        eval_mask = mask  # eval_mask = mask, during training

        batch_data["x_list"] = x  # t s n2 d
        batch_data["edge_index"] = edge_index
        batch_data["edge_weights"] = edge_weights

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        if self.scaled_target:
            target = batch.transform['y'].transform(y)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            # imputation_cyc = self._postprocess(imputation_cyc, batch_preprocessing)

        # partial loss + cycle loss
        opt.zero_grad()

        loss = self.loss_fn(imputation, target, mask)

        self.manual_backward(loss)
        opt.step()

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        # Store every randomised graphs here
        self.train_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)

        # self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        # self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # batch_data["training"] = False

        # Extract mask and target
        x = batch_data.pop("x")
        mask = batch_data.pop('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        # test = torch.sum(eval_mask, dim=(0, 1))
        # inds = torch.where(test > 0)
        y = batch_data.pop('y')

        # batch_data["x"] = x  # b s n2 d
        # batch_data["mask"] = mask  # b s n' 1
        # batch_data["known_set"] = self.known_set
        # print(self.val_set)
        adj = torch.tensor(self.adj).to(device=x.device)
        x = rearrange(x, 'b t n d -> t b n d')
        x = list(x)

        edge_index, edge_weights = dense_to_sparse(adj)

        batch_data["x_list"] = x  # t s n2 d
        batch_data["edge_index"] = edge_index
        batch_data["edge_weights"] = edge_weights

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        # trim to imputation horizon len
        # imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = batch.transform['y'].transform(y)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask.bool())

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)

        # self.val_metrics.update(imputation.detach(), y, eval_mask)
        # self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # batch_data["training"] = False

        # Extract mask and target
        x = batch_data.pop("x")
        mask = batch_data.pop('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        # test = torch.sum(eval_mask, dim=(0, 1))
        # inds = torch.where(test > 0)
        y = batch_data.pop('y')

        adj = torch.tensor(self.adj).to(device=x.device)
        x = rearrange(x, 'b t n d -> t b n d')
        x = list(x)

        edge_index, edge_weights = dense_to_sparse(adj)

        batch_data["x_list"] = x  # t s n2 d
        batch_data["edge_index"] = edge_index
        batch_data["edge_weights"] = edge_weights

        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        # self.test_metrics.update(imputation.detach(), y, eval_mask)
        # self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss