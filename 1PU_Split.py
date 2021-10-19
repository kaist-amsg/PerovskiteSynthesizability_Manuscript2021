import json
import pickle
import numpy as np
from random import shuffle
import csv
import os

nmodels = 100
valratio = 0.1
testratio = 0.1
assert nmodels%2 == 0

p = './data'

print(p)
#read
with open(p+'/id_prop.csv') as f:
    reader = csv.reader(f)
    data = sorted([row for row in reader],key=lambda x:x[0])
    
Ys = np.array([int(d[1]) for d in data])
positive = np.where(Ys==1)[0]
unlabeled = np.where(Ys==0)[0]
#randomly pick negatives
negatives = []
for _ in range(int(nmodels/2)):
    shuffle(unlabeled)
    negatives.append(np.copy(unlabeled[:len(positive)]))
    negatives.append(np.copy(unlabeled[-len(positive):]))

# pick test set
shuffle(positive)
ntest = int(np.round(len(positive)*testratio))
nval = int(np.round(len(positive)*valratio))
Splits = []
for ns in negatives:
    split = {}
    split['PTest'] = np.copy(positive[:ntest]).tolist()
    tpos = np.copy(positive[ntest:]).tolist()
    shuffle(tpos)
    split['PVal'] = np.copy(tpos[:nval]).tolist()
    split['PTrain'] = np.copy(tpos[nval:]).tolist()
    shuffle(ns)
    split['NTest'] = np.copy(ns[:ntest]).tolist()
    split['NVal'] = np.copy(ns[ntest:ntest+nval]).tolist()
    split['NTrain'] = np.copy(ns[ntest+nval:]).tolist()
    Splits.append(split)

json.dump(Splits,open('splits_Perov_All.json','w'))
