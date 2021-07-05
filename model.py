import pickle
from tqdm import trange, tqdm
import pandas as pd
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Read file
# ttv consists of [[tickerList1, dateStr1, titleStr1, triple1, vector1] ... ]
# For training, we only need vector1, and its shape is 3*100
print("Start reading file")
with open('./results/ttv.pkl', 'rb') as handle:
    f = pickle.load(handle)
train_set = []
for i in range(len(f)):
    train_set.append(f[i][4]) # vector1 is the 5th element

# set up random seed and device
seed = 2021
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print("Using device", device)

'''hyper parameters'''
config = {
    'lr': 0.000075,
    'weight_decay': 1e-5, 
    'embedding_size' : 100,
    'tensor_dim' : 100,
    'n_epochs': 200,                # maximum number of epochs
}

# A sub-model of NTN
class SubNTN(nn.Module):
    def __init__(self, embedding_size, tensor_dim):
        super(SubNTN, self).__init__()
        self.embedding_size = embedding_size
        self.tensor_dim = tensor_dim
        self.bilinear = nn.Bilinear(embedding_size, embedding_size, tensor_dim, bias=False)
        self.linear = nn.Linear(embedding_size * 2, tensor_dim)
        self.relu = nn.ReLU()
    def forward(self, e1, re):
        e1_re = torch.cat((e1, re))
        S1 = self.relu(self.bilinear(e1,re) + self.linear(e1_re))
        return S1

# Define NTN model
class NTN(nn.Module):
    def __init__(self, embedding_size, tensor_dim):
        super(NTN, self).__init__()
        self.embedding_size = embedding_size
        self.tensor_dim = tensor_dim
        self.T1 = SubNTN(embedding_size, tensor_dim)
        self.T2 = SubNTN(embedding_size, tensor_dim)
        self.T3 = SubNTN(embedding_size, tensor_dim)
        self.T4 = SubNTN(embedding_size, tensor_dim)
        self.T5 = SubNTN(embedding_size, tensor_dim)
        self.T6 = SubNTN(embedding_size, tensor_dim)
        self.linear = nn.Linear(tensor_dim, 1, bias=False) # for loss 
        
        
    def forward(self, e1, re, e2):
        # Original
        S1 = self.T1(e1,re)
        S2 = self.T2(re,e2)
        S3 = self.T4(e2,re)
        S4 = self.T5(re,e1)
        C = self.T3(S1,S2)
        Cinv = self.T6(S3,S4)
        G = self.linear(C) + self.linear(Cinv)  # G is for loss function
        return G, C, Cinv


def choose_any_word(train_set):
    # given all words in the training set, random return a word
    rand = random.randint(0,len(train_set)-1)
    out = train_set[rand]
    rand = random.randint(0,2)
    out = out[rand]
    return out
        
def train(model, config, device):
    n_epochs = config['n_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay =config['weight_decay'])
    for epoch in trange(0, n_epochs):
        model.train()                           
        losses = []
        for i in range(len(train_set)):  # triple shape: (3, 100)
            loss = 999
            cnt  = 0
            while loss > 1e-5:
                cnt += 1 
                triple = train_set[i]
                e1, re, e2 = triple[0], triple[1], triple[2]
                optimizer.zero_grad()        
                e1 = torch.from_numpy(e1).to(device)
                re = torch.from_numpy(re).to(device)
                e2 = torch.from_numpy(e2).to(device)
                corrupted = choose_any_word(train_set)
                corrupted = torch.from_numpy(corrupted).to(device)
                G, C, Cinv = model(e1, re, e2) 
                rand = random.randint(1,2)
                if rand % 2 == 0:  # corrupt e1
                    Gc, Cc, Ccinv = model(corrupted, re, e2)    
                else:  # corrupt e2
                    Gc, Cc, Ccinv = model(e1, re, corrupted)
                loss = torch.clamp(1 - G + Gc, min=0, max=None)
                loss.backward()
                optimizer.step() 
                if cnt > 10000:
                    print(i, "is not finished!:(")
                    break
           
            if i % 100 == 0:
                print("Epoch: {:03d} ==> Progress: {:05d}/{:05d}".format(epoch, i, len(train_set)))
                torch.save(model.state_dict(), './results/model.ckpt')
            

def validation(model):
    # given the trained model, return the record
    # record will be consist of many numpy arrays, each array is the concatenated result of C and Cinv
    record = []
    for i in range(len(train_set)):
        triple = train_set[i]
        e1, re, e2 = triple[0], triple[1], triple[2]
        optimizer.zero_grad() 
        e1 = torch.from_numpy(e1).to(device)
        re = torch.from_numpy(re).to(device)
        e2 = torch.from_numpy(e2).to(device)
        corrupted = choose_any_word(train_set)
        corrupted = torch.from_numpy(corrupted).to(device)
        rand = random.randint(1,2)
        G, C, Cinv = model(e1, re, e2)
        if rand % 2 == 0: # corrupt e1
            Gc, Cc, Ccinv = model(corrupted, re, e2)
        else:  # corrupt e2
            Gc, Cc, Ccinv = model(e1, re, corrupted)
        result = torch.cat((C,Cinv))
        record.append(result.cpu().detach().numpy()) # Save concated result of C and Cinv to record
    return record


'''load pickle'''
undone = []
model = NTN(embedding_size=config['embedding_size'], tensor_dim=config['tensor_dim']).to(torch.double).to(device)
train(model, config, device)
record = validation(model)
with open('./results/record.pkl', 'wb') as f:
    pickle.dump(record, f)

