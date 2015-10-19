import numpy as np
from utils import *
from sklearn.decomposition import PCA

def getDataSet(DataSetNum):
    dataFile = open('hyperplane'+str(DataSetNum)+'.arff', 'r')
    X=[]
    tags=[]
    read=False
    for line in dataFile:
        line = line.strip()
        splited= line.split(',')
        if splited[0] == '@data':
            read=True
            continue
        if not read:
            continue
        #tag = (splited[-1] == 'yes')
        #x = [v == 't' for v in splited[:-1]]
        tag= (splited[-1]=='1')
        x=  [float(v) for v in splited[:-1]]
        X.append(x)
        tags.append(tag)
    d=len(X[0])
    pca = PCA(n_components=d)
    X = pca.fit_transform(X,tags)
    #print'pca.explained_variance_ratio_', pca.explained_variance_ratio_
    return X, tags
        

 

#from sklearn.svm import SVC
#from sklearn.ensemble import AdaBoostClassifier
#clf = SVC()
#clf = AdaBoostClassifier()
from sklearn.lda import LDA
clf = LDA()
def testData(index1, index2):
    X, tags = getDataSet(index1)
    trainSize = len(X)/2
    trainX, trainTags = X[:trainSize], tags[:trainSize]
    clf.fit(trainX, trainTags)
    print index1, index1, 'train', clf.score(trainX, trainTags) 
    testX, testTags = X[trainSize:], tags[trainSize:]
    print index1, index1, 'test', clf.score(testX, testTags) 
    
    testX, testTags = getDataSet(index2)
    print index1, index2, 'test', clf.score(testX, testTags) 

testData(1, 2)
testData(2, 3)
testData(3, 4)
testData(4, 5)