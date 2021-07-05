import pickle
import numpy as np
from tqdm import tqdm, trange
import gensim.downloader as api

# download the pretrained model
wv = api.load('glove-wiki-gigaword-100') # for 100-dim model

# tagging consists of [[tickerList1, dateStr1, titleStr1, triple1], [tickerList2, dateStr2, titleStr2, triple2] ... ]
with open("./results/tagging.pkl",'rb') as f:
	data = pickle.load(f)

# result will be consist of [[tickerList1, dateStr1, title1, triple1, score] ... ]
results = []
for j in trange(len(data)):
    alist = data[j]    
    triple = data[j][3][0] # triple is the 4th element in tagging, and we choose from the first instance 
    s_value, r_value, o_value = triple['subject'], triple['relation'], triple['object']
    s_score, r_score, o_score = np.zeros(100), np.zeros(100), np.zeros(100) # change here for different dim of vector
    s_cnt, o_cnt, r_cnt = 0, 0, 0
    # iterate through all the subjects and compute average score
    for i in range(len(s_value.split())):
        try:
            s_score += wv[s_value[i]]
            s_cnt += 1
        except:
            pass
    # iterate through all the relations and compute average score
    for i in range(len(r_value.split())):
        try:
            r_score += wv[r_value[i]]
            r_cnt += 1
        except:
            pass

    # iterate thgough all the objects and compute average score
    for i in range(len(o_value.split())):
        try:
            o_score += wv[o_value[i]]
            o_cnt += 1
        except:
            pass
    # if a title is composed of {S,V,O}, we store the score 
    if s_cnt != 0 and r_cnt != 0 and o_cnt != 0:
        score = np.array([s_score/s_cnt, r_score/r_cnt, o_score/o_cnt])
        alist.append(score)
        results.append(alist)

# save the results 
with open("./results/ttv.pkl", 'wb') as f:
    pickle.dump(results, f)

print(str(len(results)) + " titles are done!")


