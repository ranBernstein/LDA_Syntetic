from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.lda import LDA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import copy
from utils import *
from sklearn.svm import SVC
from sklearn.decomposition import PCA

X=[]
y=[]
tags = set()
c = np.zeros(8)

f = open('C:/Users/ran/Downloads/shuttle.trn') 
for j,line in enumerate(f):
    splited=line.split()
    tag=(splited[-1]=='1')
    tags.add(splited[-1])
    c[int(splited[-1])]+=1
    x = [int(v) for v in splited[1:]]
    X.append(np.array(x))
    y.append(tag)
    

f = open('C:/Users/ran/Downloads/shuttle.tst') 
for j,line in enumerate(f):
    splited=line.split()
    tag=(splited[-1]=='1')
    tags.add(splited[-1])
    c[int(splited[-1])]+=1
    x = [int(v) for v in splited[1:]]
    X.append(np.array(x))
    y.append(tag)
    
print len(X)
print c