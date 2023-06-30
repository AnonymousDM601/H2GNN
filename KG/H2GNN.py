import torch
import torch.nn as nn, torch.nn.functional as F


import math 
import manifolds

import torch.nn.init as init
import numpy as np
import layers.hyp_layers  as hyp_layers

from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch.nn.init import xavier_normal_, xavier_uniform_

from geoopt import ManifoldParameter


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class H2Conv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, d=None):
        super().__init__()
        if args.manifold == 'Lorentz':
            self.manifold = getattr(manifolds, args.manifold)(1./args.c)
        self.act = nn.ReLU()
        self.linear = hyp_layers.LorentzLinear(self.manifold, in_channels, heads*out_channels, bias=True, dropout=dropout, scale=10, 
                                                fixscale=False, nonlin=None)    
        self.W = nn.Linear(in_channels, out_channels*heads, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.c = args.c
        self.args = args
        self.use_att = False
        self.drop = torch.nn.Dropout(0.2)
        self.eps = nn.Parameter(torch.tensor([0.5]))
        self.bn2 = torch.nn.BatchNorm1d(out_channels*heads)
        # self.rel_trans = nn.Parameter(torch.ones((d.num_rel(), out_channels*heads, out_channels*heads)))
        if self.use_att:
            self.key_linear = hyp_layers.LorentzLinear(self.manifold, heads*out_channels, heads*out_channels)
            self.query_linear = hyp_layers.LorentzLinear(self.manifold, heads*out_channels, heads*out_channels, nonlin='Relu')
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_channels))
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    
    def forward(self, X, emb_ty, vertex, edges, type):
        
        # linear
        N = X.shape[0]
        X = self.linear(X)

        # # agg-1st stage
        Xve = X[vertex]
        rel_trans = emb_ty[type]
        Xve = Xve - rel_trans
        Xe = scatter(Xve, edges, dim=0, reduce='sum')

        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)


        X = self.eps*Xv + X

        return X


class H2GCN(nn.Module):
    def __init__(self, dataset, n_out, c, args):
        super().__init__()
        Conv = H2Conv
        self.max_arity = 6

        self.V, self.E, self.ty = dataset.get_V_E_ty(args)
        # self.H = H
        self.act = nn.ReLU()
        self.dataset = dataset
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = args.dropout
        self.conv1 = Conv(args, n_out, n_out, heads=1, dropout=self.dropout, d=self.dataset)
        self.manifold = getattr(manifolds, args.manifold)(k=c)
        self.c = c
        self.n_out = n_out

        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.emb_E = torch.nn.Parameter(torch.randn(dataset.num_ent(), self.n_out))
        self.emb_R = torch.nn.Parameter(torch.randn(dataset.num_rel(), self.n_out))
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.emb_ty = torch.nn.Parameter(torch.randn((dataset.num_rel()-1)*6, self.n_out))

    def init(self):

        self.emb_E.data[0] = torch.ones(self.n_out)
        self.emb_R.data[0] = torch.ones(self.n_out)
        xavier_normal_(self.emb_E.data[1:])
        xavier_normal_(self.emb_R.data[1:])
        xavier_normal_(self.emb_ty.data[:])

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):

        self.E_e = self.manifold.expmap0(self.emb_E)
        self.E_e = self.conv1(self.emb_E, self.emb_ty, self.V, self.E, self.ty)
        self.E_e = self.manifold.logmap0(self.E_e)
        self.E_e[0] = torch.ones(self.n_out)
        self.R_e = self.emb_R
        self.R_e.data[0] = torch.ones(self.n_out)
        r = self.R_e[r_idx]
        e1 = self.E_e[e1_idx]
        e2 = self.E_e[e2_idx]
        e3 = self.E_e[e3_idx]
        e4 = self.E_e[e4_idx]
        e5 = self.E_e[e5_idx]
        e6 = self.E_e[e6_idx]


        x = e1 * e2 * e3 * e4 * e5 * e6 * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x
