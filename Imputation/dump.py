# def forward(self,
    #             x,
    #             mask=None,
    #             known_set=None,
    #             sub_entry_num=0,
    #             edge_weight=None,
    #             edge_features=None,
    #             training=False,
    #             reset=False,
    #             u=None,
    #             predict=False,
    #             transform=None):
    #     # x: [batches steps nodes features]
    #     # Unrandomised x, make sure this only has training nodes
    #     # x transferred here would be the imputed x
    #     # adj is the original 
    #     # mask is for nodes that need to get predicted, mask is also imputed 
    #     b, t, n, _ = x.size()
    #     x = utils.maybe_cat_exog(x, u)
    #     device = x.device

    #     o_adj = torch.tensor(self.adj).to(device)

    #     if training:
    #         o_adj = o_adj[known_set, :]
    #         o_adj = o_adj[:, known_set]

    #     edge_index, edge_weight = dense_to_sparse(o_adj)
    #     # ========================================
    #     # Simple GRU-GCN embedding
    #     # ========================================
    #     # Bigger encoders might exacerbate confounders 

    #     # flat time dimension fwd and bwd
    #     x_fwd = self.input_encoder_fwd(x)
    #     x_bwd = self.input_encoder_bwd(torch.flip(x, (1,)))

    #     # Edge encoder
    #     if self.edge_encoder is not None:
    #         assert edge_weight is None
    #         edge_weight = self.edge_encoder(edge_features)

    #     # forward encoding 
    #     x_fwd = rearrange(x_fwd, 'b t n d -> (b t) n d')
    #     out_f = x_fwd
    #     for layer in self.gcn_layers_fwd:
    #         out_f = layer(out_f, edge_index, edge_weight)
    #     out_f = out_f + self.skip_con_fwd(x_fwd)

    #     # backward encoding
    #     x_bwd = rearrange(x_bwd, 'b t n d -> (b t) n d')
    #     out_b = x_bwd
    #     for layer in self.gcn_layers_bwd:
    #         out_b = layer(out_b, edge_index, edge_weight)
    #     out_b = out_b + self.skip_con_bwd(x_bwd)

    #     # Concatenate backward and forward processes
    #     sum_out = torch.cat([out_f, out_b], dim=-1)

    #     # ========================================
    #     # Create new adjacency matrix 
    #     # ========================================
    #     # Get two graphs, one with one hop connections, another with two hop connections

    #     # TODO: need to put this in the filler
    #     if training:
    #         if reset:
    #             # d = mask.shape[-1]
    #             # sub_entry = torch.zeros(b, t, sub_entry_num, d).to(device)
    #             # mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
    #             # test = rearrange(mask, 'b t n d -> b n (t d)')
    #             # y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

    #             adj_n1, adj_n2 = self.get_new_adj(o_adj, sub_entry_num)
    #             adj_n1 = adj_n1
    #             adj_n2 = adj_n2
    #         else:
    #             if self.adj_n1 is None and self.adj_n2 is None:
    #                 adj_n1 = o_adj
    #                 adj_n2 = o_adj
    #             else:
    #                 assert self.adj_n2 is not None and self.adj_n1 is not None
    #                 adj_n1 = self.adj_n1
    #                 adj_n2 = self.adj_n2

    #         adjs = [adj_n1, adj_n2]
    #     else:
    #         adjs = [o_adj]

    #     finrecos = []
    #     finpreds = []
    #     fin_irm_all_s = []
    #     output_invars_s = []
    #     output_vars_s = []

    #     for adj in adjs:
    #         # ========================================
    #         # Calculating variant and invariant features using self-attention across different nodes using their representations
    #         # Adding the edge expansion here as well
    #         # ========================================
    #         bt, n, d = sum_out.shape
    #         if sub_entry_num != 0:
    #             sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #             full_sum_out = torch.cat([sum_out, sub_entry], dim=1)  # b*t n2 d
    #         else:
    #             full_sum_out = sum_out

    #         adj_l = dense_to_sparse(adj)

    #         t_mask = rearrange(mask, 'b t n d -> (b t) n d')
    #         full_sum_out = self.init_pass(full_sum_out, adj_l[0], adj_l[1]) * (1 - t_mask) + full_sum_out

    #         # Query represents new matrix of n1
    #         query = self.query(full_sum_out)
    #         key = self.key(full_sum_out)
    #         value = self.value(full_sum_out)

    #         # Calculate self attention between Q and K,V
    #         # Do this for the two Queries 
    #         adj_var, adj_invar, output_invars, output_vars = self.scaled_dot_product_attention(query, 
    #                                                                                            key, 
    #                                                                                            value,
    #                                                                                            mask = adj)

    #         output_invars_s.append(output_invars)
    #         output_vars_s.append(output_vars)
    #         # TODO: Check distributions after training

    #         # Edge weights must be scaled down 
    #         # Need to propagate this too
    #         # Need to scale the new edges based on the probability of the softmax
    #         # TODO: optimise this
    #         # [batch, time, node, node]

    #         # adj_invar_exp_n1 = torch.stack([adj_n1]*bt).to(device)
    #         # adj_invar_exp_n1[:, :n, :n] = adj_invar
    #         # adj_invar_exp_n2 = torch.stack([adj_n2]*bt).to(device)
    #         # adj_invar_exp_n2[:, :n, :n] = adj_invar
    #         # adj_var_exp_n1 = torch.stack([adj_n1]*bt).to(device)
    #         # adj_var_exp_n1[:, :n, :n] = adj_var
    #         # adj_var_exp_n2 = torch.stack([adj_n2]*bt).to(device)
    #         # adj_var_exp_n2[:, :n, :n] = adj_var

    #         # sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #         # rep_invars = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
    #         # rep_vars = torch.cat([output_vars, sub_entry], dim=1)  # b*t n2 d

    #         # Use this to get similarity between the known nodes 
    #         # rep_invars = (output_invars_n1 + output_invars_n2) / 2
    #         # rep_vars = (output_vars_n1 + output_vars_n2) / 2

    #         # Propagate variant and invariant features to new nodes features
    #         invar_adj = dense_to_sparse(adj_invar)
    #         var_adj = dense_to_sparse(adj_var)

    #         t_mask = rearrange(t_mask, 'b n d -> (b n) d')
    #         xh_inv = rearrange(output_invars, 'b n d -> (b n) d')
    #         xh_var = rearrange(output_vars, 'b n d -> (b n) d')

    #         # ========================================
    #         # Final prediction
    #         # ========================================
    #         # With the final representations, predict the unknown nodes again, just using the invariant features
    #         # Get reconstruction loss with pseudo labels 
    #         batches = torch.arange(0, b).to(device=device)
    #         batches = torch.repeat_interleave(batches, repeats=t*(n+sub_entry_num))

    #         finrp = self.gcn1(xh_inv, invar_adj[0], invar_adj[1], batch=batches)
    #         finrp = rearrange(finrp, '(b t n) d -> b t n d', b=b, t=t)
    #         finpreds.append(finrp)
    #         if not training and not predict:
    #             return finpreds[0]

    #         # ========================================
    #         # Reconstruction model
    #         # ========================================
    #         # Shape: b*t*n, d

    #         # rec = xh_inv * (1 - t_mask)

    #         # # Predict the real nodes by propagating back using just the invariant features
    #         # # Get reconstruction loss
    #         # finr = self.gcn1(rec, invar_adj[0], invar_adj[1], batch=batches)
    #         # finr = rearrange(finr, '(b t n) d -> b t n d', b=b, t=t)
    #         # finrecos.append(finr)

    #         # ========================================
    #         # IRM model
    #         # ========================================
    #         # Predict the real nodes by propagating back using both variant and invariant features
    #         # Get IRM loss
    #         fin_irm_all = []
        
    #         for _ in range(self.steps):
    #             rands = torch.randperm(xh_var.shape[0])
    #             rand_vars = xh_var[rands].detach()
    #             fin_irm = self.gcn2(xh_inv + rand_vars, invar_adj[0], invar_adj[1], batch=batches)
    #             fin_irm_all.append(rearrange(fin_irm, '(b t n) d -> b t n d', b=b, t=t))

    #         fin_irm_all = torch.cat(fin_irm_all)
    #         fin_irm_all_s.append(fin_irm_all)

    #     # ========================================
    #     # Size regularisation
    #     # ========================================
    #     # Regularise both graphs using CMD using 
    #     # <https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf>

    #     if training:
    #         return finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s
    #     elif predict:
    #         adj_invar = rearrange(adj_invar, '(b t) n d -> b t n d', b=b, t=t)
    #         adj_var = rearrange(adj_var, '(b t) n d -> b t n d', b=b, t=t)

    #         return finrecos, finpreds[0], adj_invar[0], adj_var[0]

    # Without attention
    # def forward(self,
    #             x,
    #             mask=None,
    #             known_set=None,
    #             sub_entry_num=0,
    #             edge_weight=None,
    #             edge_features=None,
    #             training=False,
    #             reset=False,
    #             u=None,
    #             predict=False,
    #             transform=None):
    #     # x: [batches steps nodes features]
    #     # Unrandomised x, make sure this only has training nodes
    #     # x transferred here would be the imputed x
    #     # adj is the original 
    #     # mask is for nodes that need to get predicted, mask is also imputed 
    #     b, t, n, _ = x.size()
    #     x = utils.maybe_cat_exog(x, u)
    #     device = x.device

    #     o_adj = torch.tensor(self.adj).to(device)

    #     if training:
    #         o_adj = o_adj[known_set, :]
    #         o_adj = o_adj[:, known_set]

    #     edge_index, edge_weight = dense_to_sparse(o_adj)
    #     # ========================================
    #     # Simple GRU-GCN embedding
    #     # ========================================
    #     # Bigger encoders might exacerbate confounders 

    #     # flat time dimension fwd and bwd
    #     x_fwd = self.input_encoder_fwd(x)
    #     x_bwd = self.input_encoder_bwd(torch.flip(x, (1,)))

    #     # Edge encoder
    #     if self.edge_encoder is not None:
    #         assert edge_weight is None
    #         edge_weight = self.edge_encoder(edge_features)

    #     # forward encoding 
    #     x_fwd = rearrange(x_fwd, 'b t n d -> (b t) n d')
    #     out_f = x_fwd
    #     for layer in self.gcn_layers_fwd:
    #         out_f = layer(out_f, edge_index, edge_weight)
    #     out_f = out_f + self.skip_con_fwd(x_fwd)

    #     # backward encoding
    #     x_bwd = rearrange(x_bwd, 'b t n d -> (b t) n d')
    #     out_b = x_bwd
    #     for layer in self.gcn_layers_bwd:
    #         out_b = layer(out_b, edge_index, edge_weight)
    #     out_b = out_b + self.skip_con_bwd(x_bwd)

    #     # Concatenate backward and forward processes
    #     sum_out = torch.cat([out_f, out_b], dim=-1)

    #     # ========================================
    #     # Create new adjacency matrix 
    #     # ========================================
    #     # Get two graphs, one with one hop connections, another with two hop connections

    #     # TODO: need to put this in the filler
    #     if training:
    #         if reset:
    #             # d = mask.shape[-1]
    #             # sub_entry = torch.zeros(b, t, sub_entry_num, d).to(device)
    #             # mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
    #             # test = rearrange(mask, 'b t n d -> b n (t d)')
    #             # y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

    #             adj_n1, adj_n2 = self.get_new_adj(o_adj, sub_entry_num)
    #             adj_n1 = adj_n1
    #             adj_n2 = adj_n2
    #         else:
    #             if self.adj_n1 is None and self.adj_n2 is None:
    #                 adj_n1 = o_adj
    #                 adj_n2 = o_adj
    #             else:
    #                 assert self.adj_n2 is not None and self.adj_n1 is not None
    #                 adj_n1 = self.adj_n1
    #                 adj_n2 = self.adj_n2

    #         adjs = [adj_n1, adj_n2]
    #     else:
    #         adjs = [o_adj]

    #     finrecos = []
    #     finpreds = []
    #     fin_irm_all_s = []
    #     output_invars_s = []
    #     output_vars_s = []

    #     for adj in adjs:
    #         # ========================================
    #         # Calculating variant and invariant features using self-attention across different nodes using their representations
    #         # Adding the edge expansion here as well
    #         # ========================================
    #         bt, n, d = sum_out.shape
    #         if sub_entry_num != 0:
    #             sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #             full_sum_out = torch.cat([sum_out, sub_entry], dim=1)  # b*t n2 d
    #         else:
    #             full_sum_out = sum_out

    #         adj_l = dense_to_sparse(adj)
    #         # sum_out = rearrange(sum_out, '(b t) n d -> b (t n) d', b=b, t=t)

    #         t_mask = rearrange(mask, 'b t n d -> (b t) n d')
    #         full_sum_out = self.init_pass(full_sum_out, adj_l[0], adj_l[1]) * (1 - t_mask) + full_sum_out

    #         # Query represents new matrix of n1
    #         full_sum_out = self.query(full_sum_out)
    #         # key = self.key(full_sum_out)
    #         # value = self.value(full_sum_out)

    #         # query = rearrange(query, '(b t) n d -> b n (d t)', b=b)
    #         # key = rearrange(key, '(b t) n d -> b n (d t)', b=b)
    #         # value = rearrange(value, '(b t) n d -> b n (d t)', b=b)

    #         # Calculate self attention between Q and K,V
    #         # Do this for the two Queries 
    #         # adj_var, adj_invar, output_invars, output_vars = self.scaled_dot_product_mhattention(query, 
    #         #                                                                                     key, 
    #         #                                                                                     value,
    #         #                                                                                     mask = adj,
    #         #                                                                                     n_head = t)

    #         # output_invars = rearrange(output_invars, 'b n (d t) -> (b t) n d', t=self.horizon)
    #         # output_vars = rearrange(output_vars, 'b n (d t) -> (b t) n d', t=self.horizon)
    #         # adj_invar = rearrange(adj_invar, 'b t n d -> (b t) n d')
    #         # adj_var = rearrange(adj_var, 'b t n d -> (b t) n d')


    #         output_invars = full_sum_out
    #         output_vars = full_sum_out
    #         output_invars_s.append(output_invars)
    #         output_vars_s.append(output_vars)
    #         # TODO: Check distributions after training

    #         # Edge weights must be scaled down 
    #         # Need to propagate this too
    #         # Need to scale the new edges based on the probability of the softmax
    #         # TODO: optimise this
    #         # [batch, time, node, node]

    #         # adj_invar_exp_n1 = torch.stack([adj_n1]*bt).to(device)
    #         # adj_invar_exp_n1[:, :n, :n] = adj_invar
    #         # adj_invar_exp_n2 = torch.stack([adj_n2]*bt).to(device)
    #         # adj_invar_exp_n2[:, :n, :n] = adj_invar
    #         # adj_var_exp_n1 = torch.stack([adj_n1]*bt).to(device)
    #         # adj_var_exp_n1[:, :n, :n] = adj_var
    #         # adj_var_exp_n2 = torch.stack([adj_n2]*bt).to(device)
    #         # adj_var_exp_n2[:, :n, :n] = adj_var

    #         # sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #         # rep_invars = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
    #         # rep_vars = torch.cat([output_vars, sub_entry], dim=1)  # b*t n2 d

    #         # Use this to get similarity between the known nodes 
    #         # rep_invars = (output_invars_n1 + output_invars_n2) / 2
    #         # rep_vars = (output_vars_n1 + output_vars_n2) / 2

    #         # Propagate variant and invariant features to new nodes features
    #         adj_invar = adj
    #         adj_var = adj

    #         invar_adj = dense_to_sparse(adj_invar)
    #         var_adj = dense_to_sparse(adj_var)

    #         t_mask = rearrange(t_mask, 'b n d -> (b n) d')
    #         xh_inv = rearrange(output_invars, 'b n d -> (b n) d')
    #         xh_var = rearrange(output_vars, 'b n d -> (b n) d')

    #         # ========================================
    #         # Final prediction
    #         # ========================================
    #         # With the final representations, predict the unknown nodes again, just using the invariant features
    #         # Get reconstruction loss with pseudo labels 
    #         batches = torch.arange(0, b).to(device=device)
    #         batches = torch.repeat_interleave(batches, repeats=t*(n+sub_entry_num))

    #         finrp = self.gcn1(xh_inv, invar_adj[0], invar_adj[1], batch=batches)
    #         finrp = rearrange(finrp, '(b t n) d -> b t n d', b=b, t=t)
    #         finpreds.append(finrp)
    #         if not training and not predict:
    #             return finpreds[0]

    #         # ========================================
    #         # Reconstruction model
    #         # ========================================
    #         # Shape: b*t*n, d

    #         rec = xh_inv * (1 - t_mask)
    #         fin_rec = self.init_pass(rec, invar_adj[0], invar_adj[1]) * (t_mask) + rec
    #         # fin_rec = rec
    #         # Predict the real nodes by propagating back using just the invariant features
    #         # Get reconstruction loss
    #         finr = self.gcn1(fin_rec, invar_adj[0], invar_adj[1], batch=batches)
    #         finr = rearrange(finr, '(b t n) d -> b t n d', b=b, t=t)
    #         finrecos.append(finr)

    #         # ========================================
    #         # IRM model
    #         # ========================================
    #         # Predict the real nodes by propagating back using both variant and invariant features
    #         # Get IRM loss
    #         fin_irm_all = []
        
    #         for _ in range(self.steps):
    #             rands = torch.randperm(xh_var.shape[0])
    #             rand_vars = xh_var[rands].detach()
    #             fin_irm = self.gcn2(xh_inv + rand_vars, invar_adj[0], invar_adj[1], batch=batches)
    #             fin_irm_all.append(rearrange(fin_irm, '(b t n) d -> b t n d', b=b, t=t))

    #         fin_irm_all = torch.cat(fin_irm_all)
    #         fin_irm_all_s.append(fin_irm_all)

    #     # ========================================
    #     # Size regularisation
    #     # ========================================
    #     # Regularise both graphs using CMD using 
    #     # <https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf>

    #     if training:
    #         return finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s
    #     elif predict:
    #         adj_invar = rearrange(adj_invar, '(b t) n d -> b t n d', b=b, t=t)
    #         adj_var = rearrange(adj_var, '(b t) n d -> b t n d', b=b, t=t)

    #         return finrecos[0], finpreds[0], adj_invar[0], adj_var[0]

    # Without thorough temporal disentanglement
    # def forward(self,
    #             x,
    #             mask=None,
    #             known_set=None,
    #             sub_entry_num=0,
    #             edge_weight=None,
    #             edge_features=None,
    #             training=False,
    #             reset=False,
    #             u=None,
    #             predict=False,
    #             transform=None):
    #     # x: [batches steps nodes features]
    #     # Unrandomised x, make sure this only has training nodes
    #     # x transferred here would be the imputed x
    #     # adj is the original 
    #     # mask is for nodes that need to get predicted, mask is also imputed 
    #     b, t, n, _ = x.size()
    #     x = utils.maybe_cat_exog(x, u)
    #     device = x.device

    #     o_adj = torch.tensor(self.adj).to(device)

    #     if training:
    #         o_adj = o_adj[known_set, :]
    #         o_adj = o_adj[:, known_set]

    #     edge_index, edge_weight = dense_to_sparse(o_adj)
    #     # ========================================
    #     # Simple GRU-GCN embedding
    #     # ========================================
    #     # Bigger encoders might exacerbate confounders 

    #     # flat time dimension fwd and bwd
    #     x_fwd = self.input_encoder_fwd(x)
    #     x_bwd = self.input_encoder_bwd(torch.flip(x, (1,)))

    #     # Edge encoder
    #     if self.edge_encoder is not None:
    #         assert edge_weight is None
    #         edge_weight = self.edge_encoder(edge_features)

    #     # forward encoding 
    #     x_fwd = rearrange(x_fwd, 'b t n d -> (b t) n d')
    #     out_f = x_fwd
    #     for layer in self.gcn_layers_fwd:
    #         out_f = layer(out_f, edge_index, edge_weight)
    #     out_f = out_f + self.skip_con_fwd(x_fwd)

    #     # backward encoding
    #     x_bwd = rearrange(x_bwd, 'b t n d -> (b t) n d')
    #     out_b = x_bwd
    #     for layer in self.gcn_layers_bwd:
    #         out_b = layer(out_b, edge_index, edge_weight)
    #     out_b = out_b + self.skip_con_bwd(x_bwd)

    #     # Concatenate backward and forward processes
    #     sum_out = torch.cat([out_f, out_b], dim=-1)

    #     # ========================================
    #     # Create new adjacency matrix 
    #     # ========================================
    #     # Get two graphs, one with one hop connections, another with two hop connections

    #     # TODO: need to put this in the filler
    #     if training:
    #         if reset:
    #             # d = mask.shape[-1]
    #             # sub_entry = torch.zeros(b, t, sub_entry_num, d).to(device)
    #             # mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
    #             # test = rearrange(mask, 'b t n d -> b n (t d)')
    #             # y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

    #             adj_n1, adj_n2 = self.get_new_adj(o_adj, sub_entry_num)
    #             adj_n1 = adj_n1
    #             adj_n2 = adj_n2
    #         else:
    #             if self.adj_n1 is None and self.adj_n2 is None:
    #                 adj_n1 = o_adj
    #                 adj_n2 = o_adj
    #             else:
    #                 assert self.adj_n2 is not None and self.adj_n1 is not None
    #                 adj_n1 = self.adj_n1
    #                 adj_n2 = self.adj_n2

    #         adjs = [adj_n1, adj_n2]
    #     else:
    #         adjs = [o_adj]

    #     finrecos = []
    #     finpreds = []
    #     fin_irm_all_s = []
    #     output_invars_s = []
    #     output_vars_s = []

    #     for adj in adjs:
    #         # ========================================
    #         # Calculating variant and invariant features using self-attention across different nodes using their representations
    #         # Adding the edge expansion here as well
    #         # ========================================
    #         bt, n, d = sum_out.shape
    #         if sub_entry_num != 0:
    #             sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #             full_sum_out = torch.cat([sum_out, sub_entry], dim=1)  # b*t n2 d
    #         else:
    #             full_sum_out = sum_out

    #         adj_l = dense_to_sparse(adj)
    #         # sum_out = rearrange(sum_out, '(b t) n d -> b (t n) d', b=b, t=t)

    #         t_mask = rearrange(mask, 'b t n d -> (b t) n d')
    #         full_sum_out = self.init_pass(full_sum_out, adj_l[0], adj_l[1]) * (1 - t_mask) + full_sum_out

    #         # Query represents new matrix of n1
    #         query = self.query(full_sum_out)
    #         key = self.key(full_sum_out)
    #         value = self.value(full_sum_out)

    #         query = rearrange(query, '(b t) n d -> b n (d t)', b=b)
    #         key = rearrange(key, '(b t) n d -> b n (d t)', b=b)
    #         value = rearrange(value, '(b t) n d -> b n (d t)', b=b)

    #         # Calculate self attention between Q and K,V
    #         # Do this for the two Queries 
    #         adj_var, adj_invar, output_invars, output_vars, _ = self.scaled_dot_product_mhattention(query, 
    #                                                                                             key, 
    #                                                                                             value,
    #                                                                                             mask = adj,
    #                                                                                             n_head = t)

    #         output_invars = rearrange(output_invars, 'b n (d t) -> (b t) n d', t=self.horizon)
    #         output_vars = rearrange(output_vars, 'b n (d t) -> (b t) n d', t=self.horizon)
    #         adj_invar = rearrange(adj_invar, 'b t n d -> (b t) n d')
    #         adj_var = rearrange(adj_var, 'b t n d -> (b t) n d')

    #         output_invars_s.append(output_invars)
    #         output_vars_s.append(output_vars)
    #         # TODO: Check distributions after training

    #         # Edge weights must be scaled down 
    #         # Need to propagate this too
    #         # Need to scale the new edges based on the probability of the softmax
    #         # TODO: optimise this
    #         # [batch, time, node, node]

    #         # adj_invar_exp_n1 = torch.stack([adj_n1]*bt).to(device)
    #         # adj_invar_exp_n1[:, :n, :n] = adj_invar
    #         # adj_invar_exp_n2 = torch.stack([adj_n2]*bt).to(device)
    #         # adj_invar_exp_n2[:, :n, :n] = adj_invar
    #         # adj_var_exp_n1 = torch.stack([adj_n1]*bt).to(device)
    #         # adj_var_exp_n1[:, :n, :n] = adj_var
    #         # adj_var_exp_n2 = torch.stack([adj_n2]*bt).to(device)
    #         # adj_var_exp_n2[:, :n, :n] = adj_var

    #         # sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #         # rep_invars = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
    #         # rep_vars = torch.cat([output_vars, sub_entry], dim=1)  # b*t n2 d

    #         # Use this to get similarity between the known nodes 
    #         # rep_invars = (output_invars_n1 + output_invars_n2) / 2
    #         # rep_vars = (output_vars_n1 + output_vars_n2) / 2

    #         # Propagate variant and invariant features to new nodes features
    #         invar_adj = dense_to_sparse(adj_invar)
    #         var_adj = dense_to_sparse(adj_var)

    #         t_mask = rearrange(t_mask, 'b n d -> (b n) d')
    #         xh_inv = rearrange(output_invars, 'b n d -> (b n) d')
    #         xh_var = rearrange(output_vars, 'b n d -> (b n) d')

    #         # ========================================
    #         # Final prediction
    #         # ========================================
    #         # With the final representations, predict the unknown nodes again, just using the invariant features
    #         # Get reconstruction loss with pseudo labels 
    #         batches = torch.arange(0, b).to(device=device)
    #         batches = torch.repeat_interleave(batches, repeats=t*(n+sub_entry_num))

    #         finrp = self.gcn1(xh_inv, invar_adj[0], invar_adj[1], batch=batches)
    #         finrp = rearrange(finrp, '(b t n) d -> b t n d', b=b, t=t)
    #         finpreds.append(finrp)
    #         if not training and not predict:
    #             return finpreds[0]

    #         # ========================================
    #         # Reconstruction model
    #         # ========================================
    #         # Shape: b*t*n, d

    #         rec = xh_inv * (1 - t_mask)
    #         fin_rec = self.init_pass(rec, invar_adj[0], invar_adj[1]) * (t_mask) + rec
    #         # fin_rec = rec
    #         # Predict the real nodes by propagating back using just the invariant features
    #         # Get reconstruction loss
    #         finr = self.gcn1(fin_rec, invar_adj[0], invar_adj[1], batch=batches)
    #         finr = rearrange(finr, '(b t n) d -> b t n d', b=b, t=t)
    #         finrecos.append(finr)

    #         # ========================================
    #         # IRM model
    #         # ========================================
    #         # Predict the real nodes by propagating back using both variant and invariant features
    #         # Get IRM loss
    #         fin_irm_all = []
        
    #         for _ in range(self.steps):
    #             rands = torch.randperm(xh_var.shape[0])
    #             rand_vars = xh_var[rands].detach()
    #             fin_irm = self.gcn2(xh_inv + rand_vars, invar_adj[0], invar_adj[1], batch=batches)
    #             fin_irm_all.append(rearrange(fin_irm, '(b t n) d -> b t n d', b=b, t=t))

    #         fin_irm_all = torch.cat(fin_irm_all)
    #         fin_irm_all_s.append(fin_irm_all)

    #     # ========================================
    #     # Size regularisation
    #     # ========================================
    #     # Regularise both graphs using CMD using 
    #     # <https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf>

    #     if training:
    #         return finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s
    #     elif predict:
    #         adj_invar = rearrange(adj_invar, '(b t) n d -> b t n d', b=b, t=t)
    #         adj_var = rearrange(adj_var, '(b t) n d -> b t n d', b=b, t=t)

    #         return finrecos[0], finpreds[0], adj_invar[0], adj_var[0]

    # Better version actually works
    # def forward(self,
    #             x,
    #             mask=None,
    #             known_set=None,
    #             sub_entry_num=0,
    #             edge_weight=None,
    #             edge_features=None,
    #             training=False,
    #             reset=False,
    #             u=None,
    #             predict=False,
    #             transform=None):
    #     # x: [batches steps nodes features]
    #     # Unrandomised x, make sure this only has training nodes
    #     # x transferred here would be the imputed x
    #     # adj is the original 
    #     # mask is for nodes that need to get predicted, mask is also imputed 
    #     b, t, n, _ = x.size()
    #     x = utils.maybe_cat_exog(x, u)
    #     device = x.device

    #     full_adj = torch.tensor(self.adj).to(device)

    #     o_adj = full_adj[known_set, :]
    #     o_adj = o_adj[:, known_set]

    #     edge_index, edge_weight = dense_to_sparse(o_adj)
    #     # ========================================
    #     # Simple GRU-GCN embedding
    #     # ========================================
    #     # Bigger encoders might exacerbate confounders 

    #     # flat time dimension fwd and bwd
    #     x_fwd = self.input_encoder_fwd(x)
    #     x_bwd = self.input_encoder_bwd(torch.flip(x, (1,)))

    #     # Edge encoder
    #     if self.edge_encoder is not None:
    #         assert edge_weight is None
    #         edge_weight = self.edge_encoder(edge_features)

    #     # forward encoding 
    #     x_fwd = rearrange(x_fwd, 'b t n d -> (b t) n d')
    #     out_f = x_fwd
    #     for layer in self.gcn_layers_fwd:
    #         out_f = layer(out_f, edge_index, edge_weight)
    #     out_f = out_f + self.skip_con_fwd(x_fwd)

    #     # backward encoding
    #     x_bwd = rearrange(x_bwd, 'b t n d -> (b t) n d')
    #     out_b = x_bwd
    #     for layer in self.gcn_layers_bwd:
    #         out_b = layer(out_b, edge_index, edge_weight)
    #     out_b = out_b + self.skip_con_bwd(x_bwd)

    #     # Concatenate backward and forward processes
    #     sum_out = torch.cat([out_f, out_b], dim=-1)

    #     # ========================================
    #     # Calculating variant and invariant features using self-attention 
    #     # across different nodes using their representations
    #     # ========================================
    #     # Query represents new matrix of n1
    #     query = self.query(sum_out)
    #     key = self.key(sum_out)
    #     value = self.value(sum_out)

    #     # query = rearrange(query, '(b t) n d -> b n (t d)', b=b)
    #     # key = rearrange(key, '(b t) n d -> b n (t d)', b=b)
    #     # value = rearrange(value, '(b t) n d -> b n (t d)', b=b)

    #     # # Calculate self attention between Q and K,V
    #     # # Do this for the two Queries 
    #     # adj_var, adj_invar, output_invars, \
    #     # output_vars, scores = self.scaled_dot_product_mhattention(query, 
    #     #                                                           key, 
    #     #                                                           value,
    #     #                                                           mask = o_adj,
    #     #                                                           n_head = t)
        
    #     # output_invars = rearrange(output_invars, 'b n (t d) -> (b t) n d', t=self.horizon)
    #     # output_vars = rearrange(output_vars, 'b n (t d) -> (b t) n d', t=self.horizon)
    #     # scores = rearrange(scores, 'b t n d -> (b t) n d')

    #     adj_var, adj_invar, output_invars, \
    #     output_vars, scores = self.scaled_dot_product_attention(query, 
    #                                                             key, 
    #                                                             value,
    #                                                             mask = o_adj)
        
    #     # adj_invar = rearrange(adj_invar, 'b t n d -> (b t) n d') + o_adj
    #     # adj_var = rearrange(adj_var, 'b t n d -> (b t) n d') + o_adj

    #     # ========================================
    #     # Create new adjacency matrix 
    #     # ========================================
    #     # Get two graphs, one with one hop connections, another with two hop connections
    #     if training:
    #         if reset:
    #             adj_n1, adj_n2 = self.get_new_adj(o_adj, sub_entry_num)
    #             adj_n1 = adj_n1
    #             adj_n2 = adj_n2
    #         else:
    #             if self.adj_n1 is None and self.adj_n2 is None:
    #                 adj_n1 = o_adj
    #                 adj_n2 = o_adj
    #             else:
    #                 assert self.adj_n2 is not None and self.adj_n1 is not None
    #                 adj_n1 = self.adj_n1
    #                 adj_n2 = self.adj_n2

    #         # if torch.rand(1,).item() < 0.5:
    #         #     adjs = [adj_n2]
    #         # else:   
    #         adjs = [adj_n1]
    #     else:
    #         arrange = known_set + [i for i in range(self.adj.shape[0]) if i not in known_set]
    #         n_adj = full_adj[arrange, :]
    #         n_adj = n_adj[:, arrange]

    #         adjs = [n_adj]

    #     finrecos = []
    #     finpreds = []
    #     fin_irm_all_s = []
    #     output_invars_s = []
    #     output_vars_s = []

    #     for adj in adjs:
    #         bt, n_og, d = output_invars.shape
    #         # ========================================
    #         # Edge expansion
    #         # ========================================
    #         # Propagate variant and invariant features to new nodes features
    #         new_scores = self.expand_adj(adj, scores, sub_entry_num)
    #         scores_var = new_scores.masked_fill(adj == 0, float('1e16'))
    #         scores_invar = new_scores.masked_fill(adj == 0, float('-1e16')) 

    #         # Softmax to normalize scores, producing attention weights
    #         new_var_adj = F.softmax(-scores_var, dim=-1) + adj
    #         new_var_adj = batchwise_min_max_scale(new_var_adj)

    #         new_inv_adj = F.softmax(scores_invar, dim=-1) + adj
    #         new_inv_adj = batchwise_min_max_scale(new_inv_adj)

    #         invar_adj = dense_to_sparse(new_inv_adj)
    #         var_adj = dense_to_sparse(new_var_adj)

    #         if sub_entry_num != 0:
    #             sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
    #             xh_inv = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
    #             xh_var = torch.cat([output_vars, sub_entry], dim=1)
    #         else:
    #             xh_inv = output_invars
    #             xh_var = output_vars

    #         # adj_l = dense_to_sparse(adj)
    #         # sum_out = rearrange(sum_out, '(b t) n d -> b (t n) d', b=b, t=t)

    #         # ========================================
    #         # Final prediction
    #         # ========================================
    #         t_mask = rearrange(mask, 'b t n d -> (b t n) d')
    #         # Edge weights must be scaled down 
    #         # Need to propagate this too
    #         # Need to scale the new edges based on the probability of the softmax
    #         # [batch, time, node, node]

    #         n = xh_inv.shape[1]
    #         xh_inv = rearrange(xh_inv, 'b n d -> (b n) d')
    #         xh_var = rearrange(xh_var, 'b n d -> (b n) d')
    #         # batches = torch.arange(0, b).to(device=device)
    #         # batches = torch.repeat_interleave(batches, repeats=t*n)
    #         # xh_inv = self.gcn2(xh_inv, invar_adj[0], invar_adj[1]) * (1 - t_mask) + xh_inv
    #         # xh_var = self.gcn2(xh_var, var_adj[0], var_adj[1]) * (1 - t_mask) + xh_var
    #         # xh_inv = rearrange(xh_inv, '(b t n) d -> b t n d', b=b, t=t)
    #         # xh_inv = self.layernorm2(xh_inv)
    #         # xh_inv = rearrange(xh_inv, 'b t n d -> (b t n) d')

    #         # xh_var = rearrange(xh_var, '(b t n) d -> b t n d', b=b, t=t)
    #         # xh_var = self.layernorm2(xh_var)
    #         # xh_var = rearrange(xh_var, 'b t n d -> (b t n) d')

    #         # spr_adj = dense_to_sparse(adj)
    #         xh_inv = self.gcn1(xh_inv, invar_adj[0], invar_adj[1]) * (1 - t_mask) + xh_inv
    #         xh_inv = rearrange(xh_inv, '(b t n) d -> b t n d', b=b, t=t)
    #         xh_inv = self.layernorm1(xh_inv)

    #         xh_var = self.gcn1(xh_var, var_adj[0], var_adj[1]) * (1 - t_mask) + xh_var
    #         xh_var = rearrange(xh_var, '(b t n) d -> b t n d', b=b, t=t)
    #         xh_var = self.layernorm1(xh_var)

    #         # With the final representations, predict the unknown nodes again, just using the invariant features
    #         # Get reconstruction loss with pseudo labels 

    #         # GCN
    #         # finrp = self.gcn1(xh_inv, invar_adj[0], invar_adj[1], batch=batches)

    #         finrp = self.readout1(xh_inv)
    #         finpreds.append(finrp)
    #         if not training:
    #             return finpreds[0]

    #         # ========================================
    #         # Reconstruction model
    #         # ========================================
    #         # Shape: b*t*n, d
    #         # rec = xh_inv * (1 - mask)
    #         # rec = rearrange(rec, 'b t n d -> (b t n) d')
            
    #         # # fin_rec = self.init_pass(rec, invar_adj[0], invar_adj[1]) * (t_mask) + rec
    #         # # Predict the real nodes by propagating back using just the invariant features
    #         # # Get reconstruction loss
    #         # # finr = self.gcn1(fin_rec, invar_adj[0], invar_adj[1], batch=batches)

    #         # fin_rec = self.gcn2(rec, invar_adj[0], invar_adj[1]) + rec # * (t_mask)
    #         # fin_rec = rearrange(fin_rec, '(b t n) d -> b t n d', b=b, t=t)
    #         # fin_rec = self.layernorm2(fin_rec)

    #         # finr = self.readout2(fin_rec)
    #         # finrecos.append(finr)

    #         # ========================================
    #         # MMD of embeddings
    #         # ========================================

    #         # Get embedding softmax
    #         sps_var_adj = F.softmax(-scores_var, dim=-1) * (adj + EPSILON)
    #         sps_inv_adj = F.softmax(scores_invar, dim=-1) * (adj + EPSILON)

    #         sps_var_adj = rearrange(sps_var_adj, '(b t) n d -> b t n d', b=b)
    #         sps_inv_adj = rearrange(sps_inv_adj, '(b t) n d -> b t n d', b=b)

    #         # Sample batches
    #         if self.mmd_ratio < 1.0:
    #             s_batch = int(b*self.mmd_ratio)
    #             indx = torch.randperm(b, device=device)[:s_batch]
    #             sps_inv_adj = sps_inv_adj[indx]
    #             sps_var_adj = sps_var_adj[indx]

    #             vir_inv = xh_inv[indx]
    #             vir_var = xh_var[indx]

    #         # Get most similar embedding with virtual nodes 
    #         sps_var_max = torch.argmax((sps_var_adj[:, :, :, :n_og]), dim=-1)
    #         sps_inv_max = torch.argmax((sps_inv_adj[:, :, :, :n_og]), dim=-1)

    #         # sps_var_max = sps_var_max[:, n_og:]
    #         # sps_inv_max = sps_inv_max[:, n_og:]

    #         sps_var_max = sps_var_max.unsqueeze(-1).expand(-1, -1, -1, xh_var.size(-1))
    #         sps_inv_max = sps_inv_max.unsqueeze(-1).expand(-1, -1, -1, xh_inv.size(-1))

    #         # Similar embeddings
    #         sim_inv = torch.gather(vir_inv, dim=2, index=sps_inv_max)
    #         sim_var = torch.gather(vir_var, dim=2, index=sps_var_max)

    #         # Shape: b*t, n, n
    #         emb_tru_inv = sim_inv[:, :, n_og:].view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)
    #         emb_tru_var = sim_var[:, :, n_og:].view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)

    #         emb_vir_inv = vir_inv[:, :, n_og:].view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)
    #         emb_vir_var = vir_var[:, :, n_og:].view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)

    #         finrecos.append([emb_tru_inv, emb_tru_var, emb_vir_inv, emb_vir_var])
    #         # ========================================
    #         # IRM model
    #         # ========================================
    #         # Predict the real nodes by propagating back using both variant and invariant features
    #         # Get IRM loss
    #         fin_irm_all = []
        
    #         for _ in range(self.steps):
    #             xh_var_l = rearrange(xh_var, 'b t n d -> b (t n) d', b=b, t=t)
    #             rands = torch.randperm(xh_var_l.shape[1])
    #             rand_vars = xh_var_l[:, rands, :].detach()
    #             rand_vars = rearrange(rand_vars, 'b (t n) d -> b t n d', t=t)

    #             # fin_irm = self.gcn2(xh_inv + rand_vars, invar_adj[0], invar_adj[1], batch=batches)
    #             fin_irm = self.readout3(xh_inv + rand_vars)
    #             fin_irm_all.append(fin_irm)

    #         fin_irm_all = torch.stack(fin_irm_all)
    #         fin_irm_all_s.append(fin_irm_all)
    #     # ========================================
    #     # Size regularisation
    #     # ========================================
    #     # Regularise both graphs using CMD using 
    #     # <https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf>

    #     if training:
    #         return finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s
    #     elif predict:
    #         # adj_invar = rearrange(adj_invar, '(b t) n d -> b t n d', b=b, t=t)
    #         # adj_var = rearrange(adj_var, '(b t) n d -> b t n d', b=b, t=t)

    #         # return finrecos[0], finpreds[0], adj_invar[0], adj_var[0]

    #         # new_inv_adj = rearrange(new_inv_adj, '(b t) n m -> b t n m', b=b, t=t)
    #         # new_var_adj = rearrange(new_var_adj, '(b t) n m -> b t n m', b=b, t=t)
    #         # xh_inv = rearrange(xh_inv, '(b t n) d -> b t n d', b=b, t=t)
            
    #         return finpreds[0]#, finrecos[0], new_inv_adj[0][:2], new_var_adj[0][:2], xh_inv[:2]
