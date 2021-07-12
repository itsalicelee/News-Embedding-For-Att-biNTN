import pickle
from tqdm import trange, tqdm
import pandas as pd
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import math 
# Read file
# ttv consists of [[tickerList1, dateStr1, titleStr1, triple1, vector1] ... ]
# For training, we only need vector1, and its shape is 3*100
print("Start reading file")
with open('./results/ttv.pkl', 'rb') as handle:
    f = pickle.load(handle)
train_set = []
for i in range(len(f)):
    train_set.append(f[i][4]) # vector1 is the 5th element


def getnumpy(train_set):
    e1, re, e2, corrupted = [], [], [], []
    for triple in train_set:
        e1.append(triple[0])
        re.append(triple[1])
        e2.append(triple[2])
        corrupted.append(choose_any_word(train_set))
    e1 = np.array(e1)
    re = np.array(re)
    e2 = np.array(e2)
    corrupted = np.array(corrupted)
    #print("getTensor shape:", e1.shape, re.shape, e2.shape, corrupted.shape)
    return e1, re, e2, corrupted
    
        


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
    'batch_size': 128,
    'n_epochs': 200, 
}

# self-defined dataset
class NewsDataset(Dataset):
    def __init__(self, e1, re, e2, corrupted):
        self.e1 = e1
        self.re = re
        self.e2 = e2
        self.corrupted = corrupted

    def __getitem__(self, idx):
        return self.e1[idx], self.re[idx], self.e2[idx], self.corrupted[idx]
    
    def __len__(self):
        return len(train_set)



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
        e1_re = torch.cat((e1, re), dim = 1)
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
        
def train(model, config, device, train_dataloader):
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    best_loss = 999
    for epoch in trange(0, n_epochs):
        model.train()
        for i, (e1, re, e2, corrupted) in enumerate(train_dataloader):  # triple shape: (3, 100)
            optimizer.zero_grad()        
            e1 = e1.to(device)
            re = re.to(device)
            e2 = e2.to(device)
            corrupted = corrupted.to(device)
            G, C, Cinv = model(e1, re, e2) 
            rand = random.randint(1,2)
            if rand % 2 == 0:  # corrupt e1
                Gc, Cc, Ccinv = model(corrupted, re, e2)    
            else:  # corrupt e2
                Gc, Cc, Ccinv = model(e1, re, corrupted)
            loss = torch.clamp(1 - G + Gc, min=0, max=None)
            loss = torch.flatten(loss, start_dim=0)
            loss = torch.sum(loss)/len(loss)
            loss.backward()
            optimizer.step() 
           
            if (i+1) % 20 == 0:
                print('[{:03d}/{:03d}] | Step:{:04d}/{:04d} | Train Loss: {:.4f}'.format
                        (epoch + 1, n_epochs, i+1, math.ceil(len(train_set)/batch_size), loss))
            if loss < best_loss: 
                best_loss = loss
                torch.save(model.state_dict(), './results/best_model.ckpt')
            else:
                torch.save(model.state_dict(), './results/latest_model.ckpt')
            
def validation(model):
    # given the trained model, return the record
    # record will be consist of many numpy arrays, each array is the concatenated result of C and Cinv
    record = []
    cnt = 0
    print("Start testing...")
    for i, (e1, re, e2, corrupted) in enumerate(test_dataloader):  # triple shape: (3, 100)
        optimizer.zero_grad()
        e1 = e1.to(device)
        re = re.to(device)
        e2 = e2.to(device)
        corrupted = corrupted.to(device)
        G, C, Cinv = model(e1, re, e2)
        rand = random.randint(1,2)
        if rand % 2 == 0:  # corrupt e1
            Gc, Cc, Ccinv = model(corrupted, re, e2)
        else:  # corrupt e2
            Gc, Cc, Ccinv = model(e1, re, corrupted)
        result = torch.cat((C, Cinv))
        record.append(result.cpu().detach().numpy()) # Save the result of concatenated C and Cinv to record
        cnt += 1
        
        if (cnt+1) % 50 == 0:
            print(str(cnt+1) + " titles are done!")
 
    return record



'''main'''
e1, re, e2, corrupted = getnumpy(train_set) # get e1, re, e2 corrupted as np.array
dataset = NewsDataset(e1, re, e2, corrupted) # load to dataset
train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
model = NTN(embedding_size=config['embedding_size'], tensor_dim=config['tensor_dim']).to(torch.double).to(device)
# model.load_state_dict(torch.load('./results/best_model.ckpt'))
optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay =config['weight_decay'])
train(model, config, device, train_dataloader)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
record = validation(model)
with open('./results/record.pkl', 'wb') as f:
    pickle.dump(record, f)

