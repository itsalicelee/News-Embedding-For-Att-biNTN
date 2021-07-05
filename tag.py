import pandas as pd
import numpy as np
import re
import glob
import pickle
from tqdm import tqdm, trange
from openie import StanfordOpenIE
from stanfordnlp.server import CoreNLPClient
import time

def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return sentence

print("Start reading file...")
# tdt consists of [[tickerList1, dateStr1, titleStr1], [tickerList2, dateStr2, titleStr2]...]
with open('./results/tdt.pkl','rb') as f:
    tdt = pickle.load(f)
    

# final will consist of [[tickerList1, dateStr1, titleStr1, triple1], [tickerList2, dateStr2, titleStr2, triple2]...]
final = []
print("Start tagging {subject-relation-object} relationship...")
with StanfordOpenIE() as client:
    for i in trange(len(tdt)):
        alist = tdt[i]
        title = tdt[i][2] # title is the 3nd element in each list
        triple =  client.annotate(title) # tagging relationship using OpenIE
        if triple != []:
            alist.append(triple)
            final.append(alist) 
        else:
            pass


print("Start writing file...")   
with open("./results/tagging.pkl", 'wb') as f:
    pickle.dump(final, f)

print("Original news:", len(tdt))
print("Finished tagging ", len(final), "titles!")
