import os
import numpy as np
import random
import torch
import scipy.sparse as sp
import math
import torch_sparse

class Dataset:
    def __init__(self, ds_name, max_arity=6):
        self.name = ds_name
        self.dir = os.path.join("data", ds_name)
        # THIS NEEDS TO STAY 6
        self.max_arity = max_arity
        # id zero means no entity. Entity ids start from 1.
        self.ent2id = {"":0}
        self.rel2id = {"":0}
        self.data = {}
        print("Loading the dataset {} ....".format(ds_name))
        self.data["train"] = self.read(os.path.join(self.dir, "train.txt"))
        # Shuffle the train set
        np.random.shuffle(self.data['train'])

        self.train_facts_dic = {}
        cnt = 0
        self.type = {}
        for i in range(1, self.num_rel()):
            self.type[i] = set()
        rel_cnt = np.zeros(len(self.data['train']), dtype=int)
        for ele in self.data['train']:
            self.train_facts_dic[cnt] = []
            self.type[int(ele[0])].add(cnt)
            for v in ele[1:]:
                if v != 0:
                    self.train_facts_dic[cnt].append(int(v))
            cnt = cnt + 1

        # Load the test data
        self.data["test"] = self.read_test(os.path.join(self.dir, "test.txt"))
        # Read the test files by arity, if they exist
        # If they do, then test output will be displayed by arity
        for i in range(2,self.max_arity+1):
            test_arity = "test_{}".format(i)
            file_path = os.path.join(self.dir, "test_{}.txt".format(i))
            self.data[test_arity] = self.read_test(file_path)

        self.data["valid"] = self.read(os.path.join(self.dir, "valid.txt"))
        self.batch_index = 0

    def read(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = np.zeros((len(lines), self.max_arity + 1))
        for i, line in enumerate(lines):
            tuples[i] = self.tuple2ids(line.strip().split("\t"))
        return tuples

    def read_test(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = np.zeros((len(lines),  self.max_arity + 1))
        for i, line in enumerate(lines):
            splitted = line.strip().split("\t")[1:]
            tuples[i] = self.tuple2ids(splitted)
        return tuples

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def tuple2ids(self, tuple_):
        output = np.zeros(self.max_arity + 1)
        for ind,t in enumerate(tuple_):
            if ind == 0:
                output[ind] = self.get_rel_id(t)
            else:
                output[ind] = self.get_ent_id(t)
        return output
    
    def get_V_E_ty(self, args):
        device = torch.device('cuda:'+str(args.gpu))
        indptr, indices, data = [0], [], []  # indptr[i]:前i列非零元素个数;indices:非零元素对应的行索引;data:非零元素(全为1)
        for e, vs in self.train_facts_dic.items():
            indices += vs 
            for k in range(len(vs)):
                data += [(self.data['train'][e][0]-1)*6+k]
            # data += [1] * len(vs)
            indptr.append(len(indices))
        H = sp.csc_matrix((data, indices, indptr), shape=(len(self.ent2id), len(self.train_facts_dic)), dtype=int).tocsr() # V x E，建立图关联矩阵的稀疏矩阵

        degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()  # 顶点的度，向量
        degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()  # 边的度，边数*1向量

        (row, col), value = torch_sparse.from_scipy(H) # 矩阵中不为0的元素坐标。
        row = row.to(device)
        col = col.to(device)
        value = value.to(device)
        return row, col, value

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
            print(rel)
        return self.rel2id[rel]

    def rand_ent_except(self, ent):
        # id 0 is reserved for nothing. randint should return something between zero to len of entities
        rand_ent = random.randint(1, self.num_ent() - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]):
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            ###shuffle##
            np.random.shuffle(self.data['train'])
            self.batch_index = 0
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the +1 label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the 0 arity
        return batch

    def next_batch(self, batch_size, neg_ratio, device):

        pos_batch = self.next_pos_batch(batch_size)
        batch = self.generate_neg(pos_batch, neg_ratio)

        arities = batch[:,8]
        ms = np.zeros((len(batch),6))
        bs = np.ones((len(batch), 6))
        for i in range(len(batch)):
            ms[i][0:arities[i]] = 1
            bs[i][0:arities[i]] = 0
        r  = torch.tensor(batch[:,0]).long().to(device)
        e1 = torch.tensor(batch[:,1]).long().to(device)
        e2 = torch.tensor(batch[:,2]).long().to(device)
        e3 = torch.tensor(batch[:,3]).long().to(device)
        e4 = torch.tensor(batch[:,4]).long().to(device)
        e5 = torch.tensor(batch[:,5]).long().to(device)
        e6 = torch.tensor(batch[:,6]).long().to(device)
        labels = batch[:, 7]
        ms = torch.tensor(ms).float().to(device)
        bs = torch.tensor(bs).float().to(device)
        return r, e1, e2, e3, e4, e5, e6, labels, ms, bs


    def generate_neg(self, pos_batch, neg_ratio):
        arities = [8 - (t == 0).sum() for t in pos_batch]
        pos_batch[:,-1] = arities
        neg_batch = np.concatenate([self.neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], neg_ratio) for i, c in enumerate(pos_batch)], axis=0)
        return neg_batch

    def neg_each(self, arr, arity, nr):
        arr[0,-2] = 1
        for a in range(arity):
            arr[a* nr + 1:(a + 1) * nr + 1, a + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
        return arr

    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))

