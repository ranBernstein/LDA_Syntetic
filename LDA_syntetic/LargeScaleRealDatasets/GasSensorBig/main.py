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

dic={}
alpha=1
beta=2
dic['alpha'] = alpha
dic['beta'] = beta
#T=0
R0 = 0.001
totalSize = 8378504
samplingRate = 25
numOfPeriods = 2
#periodSize = totalSize/numOfPeriods/samplingRate
L = 1500
k=1000
periodSize=min(3*L*k, totalSize/numOfPeriods/samplingRate)
#dataLengthBeforeFiltering = periodSize*k*
violationThreshold=0.95*k
nodePeriodSize = periodSize/k
d=2
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
beforeDriftP=0
beforeDriftQ=0
for _ in range(numOfPeriods/2):
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
            positive+=1
            Xp.append(x)
        else:
            Xq.append(x)
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

pred = filterOut(X)
X = np.array(X)[pred]
tags = np.array(tags)[pred]


print 'finish reading'
pca = PCA(n_components=d)
pcaInitLen = min(initLen, 100000)
pca.fit(X[:pcaInitLen], tags[:pcaInitLen])
#X_transformed = pca.transform(X)
#Xp = pca.transform(Xp)
#Xq = pca.transform(Xq)

print 'finish pca'

dataLength = len(X) 
pI = int(float(L)*positive/dataLength)
qI = L - pI
print 'dataLength', dataLength
print 'pI, qI', pI, qI
print 'len(Xp), len(Xq)', len(Xp), len(Xq)
allDataP = np.reshape(pca.transform(Xp[:k*pI]),(k,pI,d))
allDataQ = np.reshape(pca.transform(Xq[:k*qI]),(k,qI,d))

references=[]
for i in range(k): 
    referenceParams = getXYS(allDataP[i], allDataQ[i])
    references.append(referenceParams)

S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
Psize = len(allDataP[0])
Qsize = len(allDataQ[0])

syncs = []
cosines = []
timeLength = (dataLength-initLen)/k


cosines = []
leftValues = []
reals =[]
R0s =[]

accuracies=[]
adaAccuracies=[]
hits=0.0
adaHits=0.0
oracleHits=0.0
sentIndecies = set()
time=initLen/k
TVcounter = np.zeros(k+1)
NTVcounter = np.zeros(k+1)
params=[]


#params = range(initLen/k, time)
#dic['trajWindow']=trajWindow
dic['Nodes num'] = k
#dic['Window size for positive class'] = allDataP.shape[1]
#dic['Window size for negative class'] = allDataQ.shape[1]
dic['Dimension'] = d
dic['R0'] = R0
#dic['T'] = T
dic['IntancesNum'] = len(X)
dic['L'] = L
dic['Positive'] = positive
#dic['Number of syncs'] = len(syncs)
#dic['violationThreshold'] = violationThreshold
#dic['clfWindowSize'] = clfWindowSize
#dic['SentInstances'] = len(sentIndecies)
innerLoopCounter = initLen
#print 'dataLength', dataLength
print dic
while innerLoopCounter < dataLength-k+1:  
    if  innerLoopCounter%10000==0:
        print 'innerLoopCounter', innerLoopCounter
    #R0 = getR0(w0_norm, T)
    params.append(time)
    violationCounter = 0.
    leftValue = []
    for i in range(k): 
        newPoint = pca.transform(X[innerLoopCounter]).reshape((1,d))
        #newPoint =  X[innerLoopCounter].reshape((1,d))
        tag = tags[innerLoopCounter]
        if tag:
            allDataP[i] = np.concatenate((allDataP[i][1:], newPoint))
        else:
            allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint))
        #x0_i, y0_i, S0_i = references[i]
        globalParams=w0, B0, u0
        currentData=allDataP[i], allDataQ[i]
        try:
            currLeftValue, w2 = getLeftSide(references[i],  globalParams, currentData, R0*beta,alpha)
            if currLeftValue>R0:
                violationCounter += 1
        except:
            violationCounter += 1
            currLeftValue=-R0
        leftValue.append(currLeftValue)
        innerLoopCounter+=1
        
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
    leftValues.append(violationCounter/k)
    
    S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
    real = norm(w-w0)
    real /= R0
    reals.append(real)
    if real > 1 :
        TVcounter[violationCounter]+=1
    else:
        NTVcounter[violationCounter]+=1
    """   
    if real>0.1 and violationCounter==0:
        print 'leftValue', leftValue
        print real
    """
    if violationCounter >violationThreshold:
        syncs.append(time)
        for i in range(k):
            referenceParams = getXYS(allDataP[i], allDataQ[i])
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
        u0 = x0-y0 
        pca.fit(X[innerLoopCounter-pcaInitLen:innerLoopCounter], \
                tags[innerLoopCounter-pcaInitLen:innerLoopCounter])
        #X_transformed = pca.transform(X)       
    time+=1


if len(syncs) >0:
    dic['Rounds/syncs_Ratio'] = len(params)/len(syncs)
dic['RoundsPerPeriod'] = nodePeriodSize


plt.figure()
#plt.plot(params,cosines, label='True cosine simillarity')
plt.plot(params,leftValues, label='Fraction of violated nodes')
plt.plot(params,reals, label='norm(w-w0)/R0')
#plt.plot(params,R0s, label='R0')
conceptDrifts = range(nodePeriodSize, \
                      params[-1], nodePeriodSize)
for cd in conceptDrifts:
    plt.axvline(cd, color='r', label='Concept Drift')
plt.scatter(syncs, np.ones_like(syncs), c='b', label='Syncs')
#[300,600,900,1200]
#plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
#            label='Concepts Drifts', marker='x', s=100)
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Error')
plt.title(str(dic))




plt.show()