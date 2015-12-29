from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

d=20

import numpy as np
if __name__ == '__main__':
    f = open('C:/Users/ran/Downloads/Commercial/BBC.txt', 'r')
    
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    
    X=[]
    y=[]
    dataSize = 10000
    testSize=dataSize/10
    trainSize = dataSize-testSize
    for j,line in enumerate(f):
        if j>dataSize:
            break
        splited=line.split()
        tag=(splited[0]=='1')
        x=np.zeros((4125,))
        #x[:] = 0#np.NaN
        for i,feature in enumerate(splited[1:]):
            index, v = float(feature.split(':')[0]), float(feature.split(':')[1])
            x[index-1]=v
        X.append(np.array(x))
        y.append(tag)
    #imp.fit(X)
    #X = imp.transform(X)
    X = np.array(X)
    #X = X[:,np.nonzero((X!=0).sum(0))[0]]
    print len(X[0])
    pca = PCA(n_components=d)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #pcaSampleSize = min(len(X_train), 10000)
    #pca.fit(X_train[:pcaSampleSize], y[:pcaSampleSize])
    #print pca.explained_variance_ratio_
    #X = pca.transform(X)
    #X,y = np.array(X), np.array(y)
    clf =SVC()
    
    scores = cross_validation.cross_val_score(clf, X, y, n_jobs=2, cv=2)
    print scores
    #clf.fit(X[:trainSize],y[:trainSize])
    #print clf.score(X[trainSize:],y[trainSize:])
    print dataSize, 'dataSize'
    print 'd', d
