"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as func

class MemoryBank(object):
    def __init__(self, n, dim, category):
        self.n = n
        self.dim = dim
        self.category = category
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n, self.category)
        self.ptr = 0
        self.device = 'cpu'

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        n, dim = self.features.size()
        S = self.features.mm(self.features.t())  # [0, 1]
        distances, indices = torch.topk(S, topk + 1)

        features = self.features.cpu().numpy()
        distances = distances.cpu().numpy()
        indices = indices.cpu().numpy()
        
        # import faiss
        # features = self.features.cpu().numpy()
        # n, dim = features.shape[0], features.shape[1]
        # index = faiss.IndexFlatIP(dim)
        # index = faiss.index_cpu_to_all_gpus(index)
        # index.add(features)
        # distances, indices = index.search(features, topk+1) # Sample itself is included

        # evaluate
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0) 
            anchor_targets = np.expand_dims(targets, axis=1) # add dimension
            anchor_targets_ = np.repeat(anchor_targets, topk, axis=1)
            temp = np.multiply(anchor_targets_, neighbor_targets) # dot multiply
            flag_matrix = np.sum(temp, axis=2)
            flag_matrix[flag_matrix >=1] = 1

            accuracy = np.sum(flag_matrix)/ (flag_matrix.shape[0] * flag_matrix.shape[1])
            return distances, indices, accuracy

        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:1')

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    I_tr = list([])
    L_tr = list([])

    for i, (images, targets, _) in enumerate(loader):

        images = images.cuda()
        targets = targets.cuda()

        feat, _ = model(images)
        feat_ = func.normalize(feat, p=2, dim=1)
        memory_bank.update(feat_, targets)

        # add ch 20221103
        I_tr.extend(feat.cpu().data.numpy())
        L_tr.extend(targets.cpu().data.numpy())

    I_tr = np.array(I_tr)
    L_tr = np.array(L_tr)

    return I_tr, L_tr
