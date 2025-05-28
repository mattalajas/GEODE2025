import inspect
import warnings
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from active_learning import GraphStorage
from einops import rearrange
from pytorch_lightning.utilities import move_data_to_device, grad_norm
from torchmetrics import MetricCollection
from tsl import logger
from tsl.metrics.torch import MaskedMetric
from tsl.metrics.torch.functional import mre
from utils import mmd_loss, mmd_loss_single, cmd_time, cmd

warnings.filterwarnings("ignore")

NCL = 10


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]

class unKrigFillerV2(pl.LightningModule):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
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
        super(unKrigFillerV2, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn_1'], logger=False)
        self.save_hyperparameters(ignore=['loss_fn_2'], logger=False)
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.automatic_optimization = False

        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs = scheduler_kwargs

        if loss_fn is not None:
            self.loss_fn_1 = self._check_metric(loss_fn[0], on_step=True)
            self.loss_fn_2 = self._check_metric(loss_fn[1], on_step=True)
        else:
            self.loss_fn_1 = None
            self.loss_fn_2 = None

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
        return self.model(*args, **kwargs)\
    
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_mean/{name}', param.grad.mean(), on_step=True)
                self.log(f'grad_max/{name}', param.grad.max(), on_step=True)
                self.log(f'grad_min/{name}', param.grad.min(), on_step=True)      
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=True)      

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
            if res:
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
        batch_data["training"] = False

        # if self.known_set is None:
            # Get observed entries (nonzero masks across time)
        # if self.inductive:
        mask = batch_data["mask"]
        mask = rearrange(mask, "b s n 1 -> (b s) n")
        mask_sum = mask.sum(0)  # n
        known_set = torch.where(mask_sum > 0)[0]
        unknown_set = torch.where(mask_sum == 0)[0]
        arrange = torch.cat((known_set, unknown_set))
        reverse = torch.empty_like(arrange)
        reverse[arrange] = torch.arange(len(arrange)).to(arrange.device)
        arrange = arrange.detach().cpu().numpy().tolist()
        known_set = known_set.detach().cpu().numpy().tolist()

        if known_set == []:
            return None

        batch_data["sub_entry_num"] = len(unknown_set)
        batch_data["masked_set"] = unknown_set.detach().cpu().numpy().tolist()
        #     else:
        #         known_set = list(range(batch_data['mask'].shape[2]))
        # else:
        #     known_set = self.known_set

        batch_data["known_set"] = known_set
        x = batch_data["x"]
        batch_data["x"] = x[:, :, known_set, :]

        mask = batch_data["mask"]
        mask = mask[:, :, arrange, :]
        batch_data["mask"] = mask

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        batch_data.pop("edge_index", None)

        # Compute outputs and rescale
        # finpreds, _, inv_adj, var_adj, sum_out = self.predict_batch(batch, preprocess=False, postprocess=True)
        finpreds, _, _ = self.predict_batch(batch, preprocess=False, postprocess=True)
        # inv_adj = inv_adj[:, reverse, :]
        # inv_adj = inv_adj[:, :, reverse]
        # var_adj = var_adj[:, reverse, :]
        # var_adj = var_adj[:, :, reverse]

        finpreds = finpreds[:, :, reverse, :]
        mask = mask[:, :, reverse, :]

        
        output = dict(y=y,
                      y_hat=finpreds,
                      mask=mask,
                      eval_mask=eval_mask)
                    #   invars=output_invars_s[0],
                    #   vars=output_vars_s[0],
                    #   inv_adj=inv_adj,
                    #   var_adj=var_adj)
                    #   sum_out=sum_out)
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
            x = self._preprocess(x, batch_preprocessing)
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
        return y_hat, None, None

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

            y_hat, _, _ = self.predict_batch(batch, preprocess=preprocess, postprocess=postprocess)

            if isinstance(y_hat, (list, tuple)):
                y_hat = y_hat[0]

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

    def on_train_epoch_start(self) -> None:
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def configure_optimizers(self):
        cfgs = []

        for ind, val in enumerate(self.optim_class):
            cfg = dict()
            optimizer = val(self.parameters(), **self.optim_kwargs[ind])
            cfg['optimizer'] = optimizer
            if self.scheduler_class is not None:
                metric = self.scheduler_kwargs.pop('monitor', None)
                scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
                cfg['lr_scheduler'] = scheduler
                if metric is not None:
                    cfg['monitor'] = metric
            cfgs.append(cfg)

        return cfgs

class UnnamedKrigFillerV2(unKrigFillerV2):
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
                 inductive=True,
                 gradient_clip_val=None,
                 gradient_clip_algorithm=None,
                 known_set=[],
                 y1 = 1,
                 y2 = 1,
                 y3 = 1,
                 dloss = 'mmd'):
        super(UnnamedKrigFillerV2, self).__init__(model_class=model_class,
                                                  model_kwargs=model_kwargs,
                                                  optim_class=optim_class,
                                                  optim_kwargs=optim_kwargs,
                                                  loss_fn=loss_fn,
                                                  scaled_target=scaled_target,
                                                  whiten_prob=whiten_prob,
                                                  metrics=metrics,
                                                  scheduler_class=scheduler_class,
                                                  scheduler_kwargs=scheduler_kwargs)
        
        self.tradeoff = pred_loss_weight
        self.trimming = (warm_up, warm_up)

        self.known_set = known_set
        self.inductive = inductive
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

        self.train_set = known_set
        self.val_set = []
        self.e_start = True
        
        assert dloss in ['mmd', 'cmd']
        self.dloss = dloss
        
        # self.ratio = 0.

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
        storage = torch.load(filename, lambda storage, loc: storage)
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

    # def on_train_epoch_start(self):
    #     self.e_start = True

    def on_train_batch_start(self, batch, batch_idx):
        if self.e_start:
            batch_data, batch_preprocessing = self._unpack_batch(batch)

            # To make the model inductive
            # => remove unobserved entries from input data and adjacency matrix
            # if self.known_set is None:
                # Get observed entries (nonzero masks across time)
            mask = batch_data["mask"]
            mask = rearrange(mask, "b s n 1 -> (b s) n")
            mask_sum = mask.sum(0)  # n
            known_set = torch.where(mask_sum > 0)[0]
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio / 2

            dynamic_ratio = self.ratio + 0.2 * np.random.random()
            val_len = int(dynamic_ratio*len(known_set))
            arrange = torch.randperm(len(known_set), device=mask.device)
            val_set = known_set[arrange[:val_len]].detach().cpu().numpy().tolist()
            train_set = known_set[arrange[val_len:]].detach().cpu().numpy().tolist()

            self.train_set = train_set
            self.val_set = val_set
            self.e_start = False

    # Experimental
    def training_step(self, batch, batch_idx):
        # Unpack batch
        opt1, opt2 = self.optimizers()
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # To make the model inductive
        # => remove unobserved entries from input data and adjacency matrix
        # if self.known_set is None:
            # Get observed entries (nonzero masks across time)
        # if self.inductive:
        #     mask = batch_data["mask"]
        #     mask = rearrange(mask, "b s n 1 -> (b s) n")
        #     mask_sum = mask.sum(0)  # n
        #     known_set = torch.where(mask_sum > 0)[0]
        #     ratio = float(len(known_set) / mask_sum.shape[0])
        #     self.ratio = ratio

        #     dynamic_ratio = self.ratio + 0.2 * np.random.random()
        #     val_len = int(dynamic_ratio*len(known_set))
        #     arrange = torch.randperm(len(known_set), device=mask.device)
        #     val_set = known_set[arrange[:val_len]].detach().cpu().numpy().tolist()
        #     train_set = known_set[arrange[val_len:]].detach().cpu().numpy().tolist()

        #     self.train_set = train_set
        #     self.val_set = val_set
        # else:
        #     known_set = list(range(batch_data['mask'].shape[2]))
        # else:
        #     known_set = self.known_set

        train_set = self.train_set
        batch_data["known_set"] = train_set

        x = batch_data["x"]
        mask = batch_data["mask"]
        y = batch_data.pop("y")
        _ = batch_data.pop("eval_mask")  # drop this, we will re-create a new eval_mask (=mask during training)
        batch_data.pop("edge_index", None)

        # x = x[:, :, train_set, :]  # b s n1 d, n1 = num of known entries
        # mask = mask[:, :, train_set, :]  # b s n1 d
        # y = y[:, :, train_set, :]  # b s n1 d

        sub_entry_num = 0
        batch_data["reset"] = self.inductive

        # Create randomised model here
        if self.inductive:
            cur_entry_num = len(train_set)

            dynamic_ratio = self.ratio + 0.2 * np.random.random()  # ratio + 0.1
            aug_entry_num = max(int(cur_entry_num / dynamic_ratio), cur_entry_num + 1)
            sub_entry_num = aug_entry_num - cur_entry_num  # n2 - n1
            # sub_entry_num = min(aug_entry_num - cur_entry_num, mask.size(2) - cur_entry_num)  # n2 - n1
            trn_entry_num = min(int(sub_entry_num // 2), int(cur_entry_num//2)) # Max split is half, unseen nodes cannot be more than seened nodes

            assert sub_entry_num > 0, "The augmented data should have more entries than original data."
            self.sub_entry_num = sub_entry_num

        arrange = torch.randperm(cur_entry_num)
        # masked_indx = arrange[-trn_entry_num:]
        # seened_indx = arrange[:-trn_entry_num]
        t_set = torch.tensor(train_set)

        masked_indx = t_set[arrange[-trn_entry_num:]].numpy().tolist()
        seened_indx = t_set[arrange[:-trn_entry_num]].numpy().tolist()
        full = seened_indx + masked_indx
        # print(self.val_set)
        # print(sub_entry_num, cur_entry_num)
        # print(masked_indx)
        # print(seened_indx)

        # Arrange the indexes to b s (seen, masked) d
        x = x[:, :, seened_indx, :] # Input x will just be the seen nodes
        y = y[:, :, full, :]
        mask = mask[:, :, full, :]
        b, s, n, d = mask.size()

        if self.inductive:
            sub_entry = torch.zeros(b, s, sub_entry_num, d).to(x.device)
            mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
            y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

        # Mask the training masks too
        train_mask = mask.clone()
        train_mask[:, :, len(seened_indx):] = 0.
        batch_data["seened_set"] = seened_indx
        batch_data["masked_set"] = masked_indx

        eval_mask = mask  # eval_mask = mask, during training
        # eval_mask[:, :, :len(seened_indx)] = 0.

        batch_data["x"] = x  # b s seen d
        batch_data["mask"] = train_mask  # b s n' 1
        batch_data["sub_entry_num"] = sub_entry_num  # number
        batch_data["training"] = True

        # Compute predictions and compute loss
        res, _, _ = self.predict_batch(batch, preprocess=False, postprocess=False)
        finpreds, finrecos, fin_irm_all_s = res[0], res[1], res[2]
        # finrecos = torch.cat(finrecos)
        # finpreds = torch.cat(finpreds)
        # fin_irm_all_s = torch.cat(fin_irm_all_s)

        b = x.shape[0]
        if self.scaled_target:
            target = batch.transform['y'].transform(y)
        else:
            target = y
            # finrecos = self._postprocess(finrecos, batch_preprocessing)
            finpreds = self._postprocess(finpreds, batch_preprocessing)
            fin_irm_all_s = self._postprocess(fin_irm_all_s, batch_preprocessing)

        # fin_irm_all_s = rearrange(fin_irm_all_s, '(a b s) t n d -> a s b t n d', b=b, a=2)
        # fin_target = torch.cat([target, target])

        opt1.zero_grad()

        # IRM Loss
        irm_target = target[:, :, :len(train_set)]
        irm_mask = mask[:, :, :len(train_set)]
        steps = self.model.steps
        env_loss = torch.tensor([]).to(x.device)
        for i in range(steps):
            env_loss = torch.cat(
                        [env_loss,
                         self.loss_fn_1(fin_irm_all_s[i], irm_target, irm_mask.bool()).unsqueeze(0)])
        env_mean = env_loss.mean()
        env_var = torch.var(env_loss * steps)
        irm_loss = env_var + env_mean

        if self.dloss == 'mmd':
            # MMD Loss
            mmds = torch.tensor([]).to(x.device)
            for reco in finrecos:
                # MMD for inv 
                s_batch = reco.pop(0)
                mmds = torch.cat([mmds, torch.clamp(mmd_loss(reco[0], reco[2]), min=0).unsqueeze(0)])
                # mmds = torch.cat([mmds, torch.clamp(mmd_loss(reco[1], reco[3]), min=0).unsqueeze(0)])

                recon_loss = mmds.mean()

        elif self.dloss == 'cmd':
            cmds = torch.tensor([]).to(x.device)
            for reco in finrecos:
                B = reco.pop(0)
                inv_emb_tru = rearrange(reco[0], 'b t d -> (b t) d')
                inv_emb_vir = rearrange(reco[2], 'b t d -> (b t) d')

                og_nt = inv_emb_tru.size(0) // (B*s)
                cr_nt = inv_emb_vir.size(0) // (B*s)

                batches = torch.arange(0, B*s).to(device=x.device)
                og_batch = torch.repeat_interleave(batches, repeats=(og_nt))
                cr_batch = torch.repeat_interleave(batches, repeats=(cr_nt))

                cmds = torch.cat([cmds, torch.clamp(cmd(inv_emb_tru, inv_emb_vir, \
                                                        og_batch, cr_batch).mean(), min=0).unsqueeze(0)])
                # cmds = torch.cat([cmds, torch.clamp(cmd(reco[1].squeeze(0), reco[3].squeeze(0), \
                #                                         og_batch, cr_batch).mean(), min=0).unsqueeze(0)])

            recon_loss = cmds.mean()

        # mmds = torch.tensor([]).to(x.device)
        # true_embs = finrecos.pop(0)
        # for reco in finrecos:
        #     # MMD for inv 
        #     mmds = torch.cat([mmds, torch.clamp(mmd_loss(true_embs[0], reco[0]), min=0).unsqueeze(0)])
        #     mmds = torch.cat([mmds, torch.clamp(mmd_loss(true_embs[1], reco[1]), min=0).unsqueeze(0)])

        main_loss = self.loss_fn_1(finpreds, target, eval_mask.bool()) 
        loss = main_loss + self.y1 * recon_loss + self.y2 * irm_loss
        
        self.manual_backward(loss)

        if self.gradient_clip_algorithm and self.gradient_clip_val:
            self.clip_gradients(opt1, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.gradient_clip_algorithm)

        opt1.step()

        imputation = rearrange(finpreds, '(a b) t n d -> a b t n d', b=b)
        imputation = torch.mean(imputation, dim=0)
        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.log('Main loss', 
                 main_loss,                  
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False)

        self.log('Reconstruction error', 
                 recon_loss,                  
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False)
    
        self.log('IRM error', 
                 irm_loss,                  
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False)
        
        # self.log('Similarity score', 
        #          sim,                  
        #          on_step=False,
        #          on_epoch=True,
        #          logger=True,
        #          prog_bar=False)

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
        batch_data["training"] = False
        batch_data.pop("edge_index", None)
        batch_data.pop("eval_mask", None)

        # if self.known_set is None:
            # Get observed entries (nonzero masks across time)
            # if self.inductive:
                # mask = batch_data["mask"]
                # mask = rearrange(mask, "b s n 1 -> (b s) n")
                # mask_sum = mask.sum(0)  # n

        x = batch_data["x"]
        mask = batch_data["mask"]
        batch_data.pop("eval_mask", None)
        y = batch_data.pop('y')

        eval_mask = mask.clone()
        mask[:, :, self.val_set, :] = 0
        eval_mask[:, :, self.train_set, :] = 0
        x[:, :, self.val_set, :] = 0

        # if self.known_set is None:
            # Get observed entries (nonzero masks across time)
            # if self.inductive:
                # mask = batch_data["mask"]
                # mask = rearrange(mask, "b s n 1 -> (b s) n")
                # mask_sum = mask.sum(0)  # n
        known_set = self.train_set + self.val_set
        # reverse = torch.empty_like(torch.tensor(known_set))
        # reverse[known_set] = torch.arange(len(known_set))

        batch_data["x"] = x[:, :, self.train_set, :]
        batch_data["sub_entry_num"] = len(self.val_set)
        batch_data["known_set"] = self.train_set
        batch_data["masked_set"] = self.val_set
        
        mask = mask[:, :, known_set, :]
        eval_mask = eval_mask[:, :, known_set, :]
        batch_data["mask"] = mask
        y = y[:, :, known_set, :]

        # Compute predictions and compute loss
        imputation, _, _ = self.predict_batch(batch, preprocess=False, postprocess=False)
        # reverse = reverse.to(imputation.device)
        # imputation = imputation[:, :, reverse, :]

        # trim to imputation horizon len
        # imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        
        if self.scaled_target:
            target = batch.transform['y'].transform(y)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn_1(imputation, target, eval_mask)

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
        batch_data["training"] = False
        batch_data.pop("edge_index", None)
        

        # if self.known_set is None:
        #     # Get observed entries (nonzero masks across time)
        #     if self.inductive:
        mask = batch_data["mask"]
        mask = rearrange(mask, "b s n 1 -> (b s) n")
        # mask_sum = mask.sum(0)  # n
        known_set = torch.tensor(self.known_set)
        unknown_set = torch.tensor([i for i in range(mask.shape[-1]) if i not in known_set])
        arrange = torch.cat((known_set, unknown_set))
        reverse = torch.empty_like(arrange)
        reverse[arrange] = torch.arange(len(arrange)).to(arrange.device)
        arrange = arrange.detach().cpu().numpy().tolist()
        known_set = known_set.detach().cpu().numpy().tolist()

        if known_set == []:
            return None

        batch_data["sub_entry_num"] = len(unknown_set)
        #     else:
        #         known_set = list(range(batch_data['mask'].shape[2]))
        # else:
        #     known_set = self.known_set

        batch_data["known_set"] = known_set
        x = batch_data["x"]
        batch_data["x"] = x[:, :, known_set, :]
        batch_data["masked_set"] = unknown_set.detach().cpu().numpy().tolist()

        mask = batch_data["mask"]
        mask = mask[:, :, arrange, :]
        batch_data["mask"] = mask

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute outputs and rescale
        imputation, _, _ = self.predict_batch(batch, preprocess=False, postprocess=True)
        imputation = imputation[:, :, reverse, :]
        test_loss = self.loss_fn_1(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        # self.test_metrics.update(imputation.detach(), y, eval_mask)
        # self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss