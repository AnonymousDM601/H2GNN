import argparse


def parse():
    p = argparse.ArgumentParser("UniGNN: Unified Graph and Hypergraph Message Passing Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='coauthorship', help='data name (coauthorship/cocitation)')
    p.add_argument('--dataset', type=str, default='cora', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation)')
    p.add_argument('--add-self-loop', action="store_true", help='add-self-loop to hypergraph')
    p.add_argument('--activation', type=str, default='relu', help='activation layer between UniConvs')
    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=8, help='number of hidden features, note that actually it\'s #nhid x #nhead')
    p.add_argument('--nhead', type=int, default=8, help='number of conv heads')
    p.add_argument('--dropout', type=float, default=0.6, help='dropout probability after UniConv layer')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    p.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    p.add_argument('--n-runs', type=int, default=10, help='number of runs for repeated experiments')
    p.add_argument('--gpu', type=int, default=0, help='gpu id to use')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--patience', type=int, default=200, help='early stop after specific epochs')
    p.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')
    p.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')
    p.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')
    p.add_argument('--c', type=float,default=8.5)
    p.add_argument('--manifold', type=str,default='Euclidean')
    p.add_argument('--optimizer', type=str, default='Adam')
    return p.parse_args()
