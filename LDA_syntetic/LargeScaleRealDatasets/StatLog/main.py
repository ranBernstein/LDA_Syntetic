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
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    f = open('C:/Users/ran/Downloads/shuttle.trn')
    
    X=[]
    y=[]
    dataSize = 50000
    testSize=dataSize/10
    trainSize = dataSize-testSize
    for j,line in enumerate(f):
        if j>dataSize:
            break
        splited=line.split()
        tag=(splited[-1]=='1')
        #tag=int(splited[-1])
        x = [int(v) for v in splited[1:]]
        X.append(np.array(x))
        y.append(tag)
    
    clf =SVC()
    print 'Finish reading'  
    scores = cross_validation.cross_val_score(clf, X, y, n_jobs=2, cv=2)
    print scores