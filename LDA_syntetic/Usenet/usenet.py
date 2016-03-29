import numpy as np
from utils import *
from sklearn.lda import LDA
from sklearn.decomposition import PCA
import copy
font = {
    'family': 'normal',
    'weight': 'normal',
    'size': 22
    }
import matplotlib
matplotlib.rc('font', **font)

k=2
#L=100
T = 0
numOfchunks=5
chunkSize = 300
clfWindowSize = 900
initLen = clfWindowSize
violationThreshold = 0#k-1
d=2
alpha=1#1/np.sqrt(d)
#dataFile = open('usenet_recurrent3.3.data', 'r')
dataFile = open('usenet2.arff', 'r')
X=[]
tags=[]
for line in dataFile:
    line = line.strip()
    splited= line.split(',')
    #tag = (splited[-1] == 'yes')
    #x = [v == 't' for v in splited[:-1]]
    tag= (splited[-1]=='1')
    x=  [v=='1' for v in splited[:-1]]
    X.append(x)
    tags.append(tag)
dataLength = len(X)#numOfchunks*chunkSize

#X = X + X + X
pca = PCA(n_components=d)
X = pca.fit_transform(X,tags)
print(pca.explained_variance_ratio_) 





allDataP = []
allDataQ = []
for i in range(k):
    allDataP.append([])
    allDataQ.append([])
initLenTime = initLen/k
for time in range(initLenTime):
    violationCounter = 0
    for i in range(k): 
        newPoint =  X[time]
        tag = tags[time]
        if tag:
            allDataP[i].append(newPoint)
        else:
            allDataQ[i].append(newPoint)
references=[]
for i in range(k): 
    referenceParams = getXYS(allDataP[i], allDataQ[i])
    references.append(referenceParams)
allDataP = np.array(allDataP)
allDataQ = np.array(allDataQ)
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0


length = len(tags)
syncs = []
timeLength = (dataLength-initLen)/k
#params = range(initLen/k, dataLength/k)
innerLoopCounter = initLen
time=initLen/k

cosines = []
leftValues = []
reals =[]
R0s =[]

accuracies=[]
adaAccuracies=[]
oracleAccuracies=[]
hits=0.0
adaHits=0.0
oracleHits=0.0
testDataLen = dataLength - initLen
sentIndecies = set()
leftValue = 0
params=[]
while innerLoopCounter < dataLength:    
    R0 = getR0(w0_norm, T)
    params.append(time)

    
    #leftValue = 0
    violationCounter = 0
    leftValue = []
    for i in range(k): 
        if innerLoopCounter >= dataLength:
            break
        newPoint =  X[innerLoopCounter].reshape((1,d))
        tag = tags[innerLoopCounter]
        if tag:
            allDataP[i] = np.concatenate((allDataP[i][1:], newPoint))
        else:
            allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint))

        globalParams=w0, B0, u0
        currentData=allDataP[i], allDataQ[i]
        try:
            currLeftValue, w2 = getLeftSide(references[i],  globalParams, currentData, R0,alpha)
            if currLeftValue>R0:
                violationCounter += 1
        except:
            violationCounter += 1
            currLeftValue=-R0
        leftValue.append(currLeftValue)
        windowIndex = innerLoopCounter%clfWindowSize
        
        innerLoopCounter+=1
    leftValue = np.max(leftValue)
    leftValue /= R0
    leftValues.append(leftValue)
    
    S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
    real = norm(w-w0)
    real /= R0
    reals.append(real)      
    
    
    if violationCounter >violationThreshold:
    #if real>R0:
        syncs.append(time+1)
        for i in range(k):
            referenceParams = getXYS(allDataP[i], allDataQ[i])
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
        u0 = x0-y0
        sentIndecies= sentIndecies | set(range(innerLoopCounter-clfWindowSize,innerLoopCounter))
    
        params.append(time)
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
        real = norm(w-w0)
        real /= R0
        reals.append(real)
        leftValue = []
        for i in range(k): 
            #x0_i, y0_i, S0_i = references[i]
            #localParams=S0_i, x0_i, y0_i
            globalParams=w0, B0, u0
            currentData=allDataP[i], allDataQ[i]
            currLeftValue, waste = getLeftSide(references[i],  globalParams, 
                                               currentData, R0, alpha)
            leftValue.append(currLeftValue)
        leftValue = np.max(leftValue)
        leftValue /= R0
        leftValues.append(leftValue)

    time+=1
    
print syncs
print len(syncs)

dic={}
dic['Nodes num'] = k
#dic['Window size for positive class'] = allDataP.shape[1]
#dic['Window size for negative class'] = allDataQ.shape[1]
dic['BoundWindowSize'] = allDataP.shape[1] + allDataQ.shape[1]
dic['alpha'] = alpha
dic['Dimension'] = d
dic['Threshold'] = T
#dic['Number of syncs'] = len(syncs)
dic['violationThreshold'] = violationThreshold
dic['clfWindowSize'] = clfWindowSize
dic['Rounds/syncs_Ratio'] = len(params)/len(syncs)
#dic['SentInstances'] = len(sentIndecies)
"""
plt.title(str(dic))
plt.plot(params, accuracies, label='Accuracy '+str(hits/testDataLen))
plt.plot(params, adaAccuracies, label='Adaptive accuracy '+str(adaHits/testDataLen))
#plt.plot(params, oracleAccuracies, label='Oracle accuracy '+str(oracleHits/testDataLen))
plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
            label='Concepts Drifts', marker='x', s=100)
for sync in syncs:
    plt.axvline(sync, color='r')
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Accuracy')

plt.figure()
"""
#plt.plot(params,cosines, label='True cosine simillarity')
plt.plot(params,leftValues, label='DLDA Bound', c='b')
plt.plot(params,reals, label='norm(w-w0)', c='g', linestyle='--')
#plt.plot(params,R0s, label='R0')
conceptsDrifts = [chunkSize*i/k for i in range(1, numOfchunks)]
first=True
for conceptsDrift in conceptsDrifts:
    if first:
        first=False
        plt.axvline(conceptsDrift, linestyle='dotted', color='r', label='Concepts Drifts')
    else:
        plt.axvline(conceptsDrift, linestyle='dotted', color='r')
#plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
#            label='Concepts Drifts', marker='x', s=100)
first=True
for sync in syncs:
    if first:
        first=False
        plt.axvline(sync, linestyle='dashed', color='b', label='Syncs')
    else:
        plt.axvline(sync,  linestyle='dashed', color='b')
#plt.scatter(syncs, np.ones_like(syncs), c='b', label='Syncs')
#[300,600,900,1200]
#plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
#            label='Concepts Drifts', marker='x', s=100)
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Model Drift')
#plt.title(str(dic))

plt.show() 
