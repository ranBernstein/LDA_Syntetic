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

dic={}
alpha=1
T=0.5
totalSize = 8378504
samplingRate = 10000
numOfPeriods = 2
periodSize = totalSize/numOfPeriods/samplingRate
k=24
violationThreshold=k-1#0.9*k
nodePeriodSize = periodSize/k
d=2
L = 5000
initLen = L*k
ethylene_CO = open('C:/Users/ran/Downloads/ethylene_CO.txt')
ethylene_methane = open('C:/Users/ran/Downloads/ethylene_methane.txt')
headers = ethylene_CO.readline()
headers = ethylene_methane.readline()

X=[]
tags=[]
y1=[]
y2=[]
positive=0
Xp=[]
Xq=[]
pBegin=0
qBegin=0
for periodNum in range(numOfPeriods/2):
    for j,line in enumerate(ethylene_CO):
        if j>periodSize*samplingRate:
            break
        if j%samplingRate != 0:
            continue
        splited = line.split()
        x = [float(v) for v in splited[3:]]
        eth = float(splited[1])
        other = float(splited[2])
        #tags.append(bool(other>0) ^ bool(eth>0))
        tag = bool(other>0)
        if tag:
            if periodNum==0:
                pBegin+=1
            positive+=1
            Xp.append(x)
        else:
            Xq.append(x)
            if periodNum==0:
                qBegin+=1
        tags.append(tag)
        y1.append(eth)
        y2.append(other)
        X.append(np.array(x))
        
    for j,line in enumerate(ethylene_methane):
        if j>periodSize*samplingRate:
            break
        if j%samplingRate != 0:
            continue
        splited = line.split()
        eth = float(splited[1])
        other = float(splited[2])
        #tags.append(bool(other>0) ^ bool(eth>0))
        tag = bool(other>0)
        if tag:
            positive+=1
            Xp.append(x)
        else:
            Xq.append(x)
        tags.append(tag)
        y1.append(eth)
        y2.append(float(splited[2]))
        x = [float(v) for v in splited[3:]]
        X.append(np.array(x))

print 'finish raeding'

from sklearn import svm
from scipy import stats
from sklearn.covariance import EllipticEnvelope
def filterOut(x):
    x = np.array(x)
    outliers_fraction=0.05
    #clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,  kernel="rbf", gamma=0.1) 
    clf = EllipticEnvelope(contamination=.1)
    clf.fit(x)
    y_pred = clf.decision_function(x).ravel()
    threshold = stats.scoreatpercentile(y_pred,
                                        100 * outliers_fraction)
    y_pred = y_pred > threshold
    x = x[y_pred]
    return y_pred
pred = filterOut(X)
X = np.array(X)[pred]
tags = np.array(tags)[pred]
Xq = np.array(Xq)[filterOut(Xq)]
Xp = np.array(Xp)[filterOut(Xp)]

pca = PCA(n_components=d)
pca.fit(X[:initLen], tags[:initLen])
X = pca.transform(X)
Xp = pca.transform(Xp)
Xq = pca.transform(Xq)

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

ax1.scatter(Xp[:pBegin,0], Xp[:pBegin,1], c='b', label='CO before')
ax1.scatter(Xq[:qBegin,0], Xq[:qBegin,1], c='red', label='No CO before')
ax1.legend().draggable()
ax1.set_title('A')
ax2.scatter(Xp[pBegin:,0], Xp[pBegin:,1], c='green', label='Methane after')
ax2.scatter(Xq[qBegin:,0], Xq[qBegin:,1], c='yellow', label='No Methane after')
ax2.legend().draggable()
ax2.set_title('B')
ax2.set_ylabel('Second principal component')
from sklearn.lda import LDA

clf = LDA()
clf.fit(X[:len(X)/2], tags[:len(X)/2])
w = clf.coef_[0]
w1 = clf.coef_[1][0]  
a = -w[0] / w1
xx = np.linspace(-10000, 15000) 
yy = a * xx - (clf.intercept_[0]) / w1

ax3.plot(xx, yy, label='Before separator')
ax3.scatter(Xp[:pBegin,0], Xp[:pBegin,1], label='CO (before)', c='b')
ax3.scatter(Xq[:qBegin,0], Xq[:qBegin,1], label='No CO (before)', c='red')
ax3.set_xlabel('First principal component')
ax3.set_title('C')

#plt.figure()

clf.fit(X[len(X)/2:], tags[len(X)/2:])
w = clf.coef_[0]
w1 = clf.coef_[1][0]  
a = -w[0] / w1
yy = a * xx - (clf.intercept_[0]) / w1
ax3.plot(xx, yy, label='After separator')
ax3.scatter(Xp[pBegin:,0], Xp[pBegin:,1], label='Methane (after)', c='green')
ax3.scatter(Xq[qBegin:,0], Xq[qBegin:,1], label='No Methane (after)', c='yellow')
ax3.legend().draggable()



plt.show()
