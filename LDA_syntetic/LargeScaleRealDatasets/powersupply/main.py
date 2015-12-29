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
    f = open('C:/Users/ran/Downloads/powersupply.arff')
    
    X=[]
    y=[]
    dataSize = 50000
    testSize=dataSize/10
    trainSize = dataSize-testSize
    for j,line in enumerate(f):
        if j>dataSize:
            break
        if '\n' in line:
            line=line[:-1]
            splited=line.split(',')
        try:
            hour = int(splited[2])
        except:
            print splited
        tag=(hour > 7 and hour < 19)
        #tag=int(splited[-1])
        x = [float(v) for v in splited[:2]]
        X.append(np.array(x))
        y.append(tag)
    
    clf =SVC()
    print 'Finish reading'  
    scores = cross_validation.cross_val_score(clf, X, y, n_jobs=2, cv=2)
    print scores