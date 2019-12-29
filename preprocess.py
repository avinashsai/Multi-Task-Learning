import os
import re
import sys
import string
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,StratifiedKFold

datapath = 'fudan-mtl-dataset/'
domains = []
for files in os.listdir(datapath):
    if(files.split('.')[-1]=='train'):
        domains.append(files.split('.')[0])


for domain in domains:
    trainfile = domain+'.task.train'
    testfile = domain+'.task.test'
    poscorpus = []
    negcorpus = []
    poscount = 0
    negcount = 0
    with open(datapath+trainfile,'r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            lab = int(words[0])
            sen = " ".join(word for word in words[1:-1])
            if(lab):
                poscorpus.append(sen)
                poscount+=1
            else:
                negcorpus.append(sen)
                negcount+=1
                
    testdata = []
    testlab = []
    with open(datapath+testfile,'r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            lab = int(words[0])
            sen = " ".join(word for word in words[1:-1])
            testdata.append(sen)
            testlab.append(lab)
    
    valdata = poscorpus[-100:] + negcorpus[-100:]
    traindata = poscorpus[:-100] + negcorpus[:-100]
    vallab = np.zeros(200)
    vallab[:100] = 1
    trainlab = np.zeros(len(traindata))
    trainlab[0:poscount] = 1
    print("Domain {} Training Size {} Validation Size {} Testing Size {} ".format(domain,len(traindata),len(valdata),
                                                                                  len(testdata)))
    traindata,trainlab = shuffle(traindata,trainlab)
    valdata,vallab = shuffle(valdata,vallab)
    traindf = pd.DataFrame({'text':traindata,'label':trainlab})
    valdf = pd.DataFrame({'text':valdata,'label':vallab})
    testdf = pd.DataFrame({'text':testdata,'label':testlab})
    
    directory = 'mtl/'+domain.lower()
    if os.path.isdir(directory) is False:
        os.makedirs(directory)
    traindf.to_csv(directory+'/train.csv',index=False,header=True)
    valdf.to_csv(directory+'/val.csv',index=False,header=True)
    testdf.to_csv(directory+'/test.csv',index=False,header=True)