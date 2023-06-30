import torch
import torch.nn as nn, torch.nn.functional as F


import math 
import manifolds

import torch.nn.init as init
import numpy as np
import layers.hyp_layers  as hyp_layers

from torch_scatter import scatter
from torch_geometric.utils import softmax

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

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        if args.manifold == 'Lorentz':
            self.manifold = getattr(manifolds, args.manifold)(1./args.c)
        self.linear = hyp_layers.LorentzLinear(self.manifold, in_channels, heads*out_channels, bias=False, dropout=dropout, scale=10, 
                                                fixscale=False, nonlin=None)    
        self.W = nn.Linear(in_channels, out_channels*heads, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.c = args.c
        self.args = args
        self.eps = ManifoldParameter(torch.tensor([0.5]))
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        
        # linear
        N = X.shape[0]
        X = self.linear(X)

        # agg-1st stage
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='sum')
        denom = (-self.manifold.inner(None, Xe, keepdim=True))       # -Xe与自身的Minkowski inner product.
        denom = denom.abs().clamp_min(1e-8).sqrt()    # 绝对值，最小值设为1e-8,开平方
        Xe = Xe / (denom * math.sqrt(1.0/self.c))     # 归一化？

        # agg-2nd stage
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        # denom = (-self.manifold.inner(None, Xv, keepdim=True))
        # denom = denom.abs().clamp_min(1e-8).sqrt()
        # Xv = Xv / (denom * math.sqrt(1.0/self.c)) 

        X = self.eps*Xv+X

        # NOTE: skip concat here?

        return X



class H2GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E, c):
        """H2GNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = H2Conv
        self.manifold_name = args.manifold
        if self.manifold_name in ['Lorentz', 'Hyperboloid']:  
            nfeat = nfeat + 1
            nhid = nhid + 1 
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.dropout)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.dropout) for _ in range(nlayer-2)]
        )
        self.V = V 
        self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = c
        if self.manifold.name == 'Lorentz':
            self.cls = ManifoldParameter(self.manifold.random_normal((nclass, nhid*nhead), std=1./math.sqrt(nhid*nhead)), manifold=self.manifold)
        else:
            self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)

    def forward(self, X):
        V, E = self.V, self.E

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(X)
            X = torch.cat([o[:, 0:1], X], dim=1)
        if self.manifold.name == 'Lorentz':
            X = self.manifold.expmap0(X)
        else:
            X_tan = self.manifold.proj_tan0(X, self.c)
            X_hyp = self.manifold.expmap0(X_tan, c=self.c)
            X = self.manifold.proj(X_hyp, c=self.c)

        
        for conv in self.convs:
            X = conv(X, V, E)
            if self.manifold_name != 'Lorentz':
                xt = self.act(self.manifold.logmap0(X, c=self.c))
                xt = self.manifold.proj_tan0(xt, c=self.c)
                X = self.manifold.proj(self.manifold.expmap0(xt, c=self.c), c=self.c)
            X = self.dropout(X)
            

        if self.manifold.name == 'Lorentz':
            X = (2 + 2 * self.manifold.cinner(X, self.cls))
        else:
            X = self.conv_out(X, V, E)
            X = self.manifold.proj_tan0(self.manifold.logmap0(X, c=self.c), c=self.c)
        return F.log_softmax(X, dim=1)