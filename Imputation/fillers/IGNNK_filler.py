import inspect
import warnings
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from active_learning import GraphStorage
from einops import rearrange
from KITS_filler import Filler
from pytorch_lightning.utilities import move_data_to_device
from other_exp_utils import load_metr_la_rdata, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,test_error
from tsl import logger
import random

warnings.filterwarnings("ignore")

NCL = 10


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


class GCNCycVirtualFiller(Filler):
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
                 gradient_clip_val=None,
                 gradient_clip_algorithm=None,
                 known_nodes=None,
                 individual_reg=0,
                 n_o_n_m=150,
                 max_value=50,
                 val_ratio=0.1):
        super(GCNCycVirtualFiller, self).__init__(model_class=model_class,
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
        self.n_o_m_n = n_o_n_m
        self.max_value = max_value
        self.known_set = None
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        self.cur_epo = -1
        self.indiv_reg = individual_reg

        self.train_set = known_nodes[:-1]
        self.val_set = [known_nodes[-1]]
        self.e_start = True

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

    def training_step(self, batch, batch_idx):
        # Unpack batch
        opt = self.optimizers()
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        train_set = self.train_set
        batch_data["known_set"] = train_set

        x = batch_data["x"]
        mask = batch_data["mask"]
        y = batch_data.pop("y")
        _ = batch_data.pop("eval_mask")  # drop this, we will re-create a new eval_mask (=mask during training)

        x = x[:, 0, train_set, :]  # b s n1 d, n1 = num of observed entries
        mask = mask[:, 0, train_set, :]  # b s n1 d
        y = y[:, 0, train_set, :]  # b s n1 d
        og_adj = self.model.adj.clone().to(device=x.device)
        A_s = og_adj[:, train_set]
        A_s = A_s[train_set, :]

        b,t,n,_ = x.shape
        batch_data["reset"] = self.inductive

        t_random = np.random.randint(0, high=(b - t), size=b, dtype='l')
        know_mask = set(random.sample(range(0,n), self.n_o_m_n)) #sample n_o + n_m nodes
        feed_batch = []
        for j in range(b):
            feed_batch.append(x[t_random[j]: t_random[j] + t, :][:, list(know_mask)]) #generate 8 time batches
        
        inputs = torch.stack(feed_batch).to(x.device)
        inputs_omask = torch.ones_like(inputs).to(x.device)
        inputs_omask[inputs == 0] = 0           # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
                                                # For other datasets, it is not necessary to mask 0 values
                                                
        missing_index = torch.ones_like(inputs).to(x.device)
        for j in range(b):
            missing_mask = random.sample(range(0,self.n_o_m_n),len(self.val_set)) #Masked locations
            missing_index[j, :, missing_mask] = 0
            
        Mf_inputs = inputs * inputs_omask * missing_index / self.max_value #normalize the value according to experience
        mask = inputs_omask  #The reconstruction errors on irregular 0s are not used for training
        
        A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]   #Obtain the dynamic adjacent matrix
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))
        
        y = inputs/self.max_value #The label
        # X, A_q, A_h
        batch_data["X"] = Mf_inputs  # b s n2 d
        batch_data["A_q"] = A_q  # number
        batch_data["A_h"] = A_h

        # Compute predictions and compute loss
        X_res = self.predict_batch(batch, preprocess=False, postprocess=False)

        if self.scaled_target:
            target = batch.transform['y'].transform(y)
        else:
            target = y
            X_res = self._postprocess(X_res, batch_preprocessing)

        # partial loss + cycle loss
        opt.zero_grad()

        loss = self.loss_fn(X_res, target, mask)
        
        self.manual_backward(loss)

        if self.gradient_clip_algorithm and self.gradient_clip_val:
            self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.gradient_clip_algorithm)

        opt.step()

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        # Store every randomised graphs here
        self.train_metrics.update(imputation.detach(), y, mask)
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

        # Extract mask and target
        x = batch_data["x"]
        mask = batch_data.get('mask')
        _ = batch_data.pop('eval_mask', None)
        # test = torch.sum(eval_mask, dim=(0, 1))
        # inds = torch.where(test > 0)
        y = batch_data.pop('y')
    
        eval_mask = mask.clone()
        mask[:, :, self.val_set, :] = 0
        eval_mask[:, :, self.train_set, :] = 0
        x[:, :, self.val_set, :] = 0

        known_set = self.train_set + self.val_set
        x = x[:, :, known_set, :]  # b s n1 d, n1 = num of observed entries
        mask = mask[:, :, known_set, :]  # b s n1 d
        eval_mask = eval_mask[:, :, known_set, :]
        y = y[:, :, known_set, :]  # b s n1 d

        batch_data["x"] = x  # b s n2 d
        batch_data["mask"] = mask  # b s n' 1
        batch_data["known_set"] = known_set
        # print(self.val_set)

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

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
        batch_data["training"] = False

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

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