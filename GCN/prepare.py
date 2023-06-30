import os

import config

from model import *
import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import optimizers

args = config.parse()

args.device = 'cuda:' + str(args.gpu) if int(args.gpu) >= 0 else 'cpu'

def accuracy(Z, Y):
    
    return 100 * Z.argmax(1).eq(Y).float().mean().item()


import torch_sparse

def fetch_data(args):
    from data import data 
    dataset, _, _ = data.load(args)
    args.dataset_dict = dataset 

    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
   
    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    X, Y = X.to(args.device), Y.to(args.device)
    return X, Y, G       # X:正则化之后的特征矩阵  Y:一维tensor，每个节点属于第几类  G：节点和边的从属关系，边：{节点}

def  initialise(X, Y, G, args, unseen=None):
    """
    initialises model, optimiser, normalises graph, and features
    
    arguments:
    X, Y, G: the entire dataset (with graph, features, labels)
    X:features
    Y:labels
    G:graph
    args: arguments
    unseen: if not None, remove these nodes from hypergraphs

    returns:
    a tuple with model details (model, optimiser)    
    """
    
    G = G.copy()  # 防止改变原对象
    print(G)
    
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            G[f'self-loop-{v}'] = [v]



    N, M = X.shape[0], len(G)  # N:节点数 M:边数
    indptr, indices, data = [0], [], []  # indptr[i]:前i列非零元素个数;indices:非零元素对应的行索引;data:非零元素(全为1)
    for e, vs in G.items():
        indices += vs 
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr() # V x E，建立图关联矩阵的稀疏矩阵

    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    degE = scatter(degV[V], E, dim=0, reduce='sum') # 聚合得到边的度
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    # degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge


    V, E = V.to(args.device), E.to(args.device)
    args.degV = degV.to(args.device)
    args.degE = degE.to(args.device)
    args.degE2 = degE2.pow(-1.).to(args.device)


    


    nfeat, nclass = X.shape[1], len(Y.unique()) # 特征维度，类别数
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead
    c = args.c
    # H = torch.FloatTensor(np.array(H.todense()))
    # H = H.to(args.device)

    # H2GNN and optimiser
    model = H2GCN(args, nfeat, nhid, nclass, nlayer, nhead, V, E, c)
    opt = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=5e-4)


    model.to(args.device)
   
    return model, opt



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
