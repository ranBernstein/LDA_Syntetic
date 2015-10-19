import numpy as np
from utils import *
from sklearn.lda import LDA
from sklearn.decomposition import PCA
import copy

k=1
#L=100
T = 0
periodSize = 300

numOfchunks=5
chunkSize = 300
clfWindowSize = 100
initLen = clfWindowSize
violationThreshold = k-1
d=2

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
pca = PCA(n_components=d)
X = pca.fit_transform(X,tags)
print(pca.explained_variance_ratio_) 

clf = LDA()
clf.fit(X, tags)
print clf.score(X, tags) 

startIndex = 0
endIndex = startIndex + periodSize
currX = X[startIndex:endIndex]
currY = tags[startIndex:endIndex]
clf.fit(currX, currY)
print startIndex, endIndex, clf.score(currX, currY) 

startIndex += periodSize 
endIndex += periodSize
currX = X[startIndex:endIndex]
currY = tags[startIndex:endIndex]
print startIndex, endIndex, clf.score(currX, currY) 

startIndex += periodSize 
endIndex += periodSize
currX = X[startIndex:endIndex]
currY = tags[startIndex:endIndex]
print startIndex, endIndex, clf.score(currX, currY) 

startIndex += periodSize 
endIndex += periodSize
currX = X[startIndex:endIndex]
currY = tags[startIndex:endIndex]
print startIndex, endIndex, clf.score(currX, currY) 

startIndex += periodSize 
endIndex += periodSize
currX = X[startIndex:endIndex]
currY = tags[startIndex:endIndex]
print startIndex, endIndex, clf.score(currX, currY) 



allDataP = []
allDataQ = []
for i in range(k):
    allDataP.append([])
    allDataQ.append([])
initLenTime = initLen/k
for time in range(initLen):
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
S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)

clf = LDA()
clf.fit(X[:initLen], tags[:initLen])
adaClf = LDA()
adaClf.fit(X[:initLen], tags[:initLen])
oracleClf = LDA()
oracleClf.fit(X[:initLen], tags[:initLen])

lastRes = {}
pred = clf.predict(X[initLen-clfWindowSize:initLen])
real =  tags[initLen-clfWindowSize:initLen]
lastRes = np.logical_not(np.logical_xor(pred, real))
adaLastRes = copy.copy(lastRes)
oracleLastRes = copy.copy(lastRes)
dataLength = numOfchunks*chunkSize

length = len(tags)
syncs = []
timeLength = (dataLength-initLen)/k
params = range(initLen/k, dataLength/k)
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
while innerLoopCounter < dataLength:    
    R0 = getR0(w0_norm, T)
    
    S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
    real = norm(w-w0)
    reals.append(real)
    leftValue = 0
    for i in range(k): 
        x0_i, y0_i, S0_i = references[i]
        currLeftValue = getLeftSide(S0_i, x0_i, y0_i, w0_norm, \
            B0_inverted, allDataP[i], allDataQ[i], T)
        leftValue+=currLeftValue
    leftValue /= k
    leftValues.append(leftValue)
    
    c = cosineSimilarity(w0, w) 
    cosines.append(c)
    R0s.append(R0)
    
    violationCounter = 0
    for i in range(k): 
        newPoint =  X[innerLoopCounter].reshape((1,d))
        tag = tags[innerLoopCounter]
        if tag:
            allDataP[i] = np.concatenate((allDataP[i][1:], newPoint))
        else:
            allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint))

        x0_i, y0_i, S0_i = references[i]
        currLeftValue = getLeftSide(S0_i, x0_i, y0_i, w0_norm, \
            B0_inverted, allDataP[i], allDataQ[i], T)
        #leftValue+=currLeftValue
        isInSafeZone = (currLeftValue<=R0)
        if not isInSafeZone:
            violationCounter += 1
        
        windowIndex = innerLoopCounter%clfWindowSize
        res = clf.predict([X[innerLoopCounter]])
        lastRes[windowIndex] = (res==tag)
        hits+=(res==tag)
        
        adaRes = adaClf.predict([X[innerLoopCounter]])
        adaLastRes[windowIndex] = (adaRes==tag)
        adaHits+=(adaRes==tag)
        
        relevantRange = innerLoopCounter%chunkSize
        if relevantRange>5:
            oracleClf.fit(X[innerLoopCounter-relevantRange:innerLoopCounter], 
                      tags[innerLoopCounter-relevantRange:innerLoopCounter])
        oracleRes= oracleClf.predict([X[innerLoopCounter]])
        oracleLastRes[windowIndex] = (oracleRes==tag)
        oracleHits+=(oracleRes==tag)
        innerLoopCounter+=1
    
    
    accuracy = float(np.sum(lastRes))/clfWindowSize
    accuracies.append(accuracy)
    adaAccuracy = float(np.sum(adaLastRes))/clfWindowSize
    adaAccuracies.append(adaAccuracy)
    oracleAccuracies.append(float(np.sum(oracleLastRes))/clfWindowSize)
    
    
    
    if violationCounter >violationThreshold:
    #if real>R0:
        syncs.append(time+1)
        for i in range(k):
            referenceParams = getXYS(allDataP[i], allDataQ[i])
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
        
        adaClf.fit(X[innerLoopCounter-clfWindowSize:innerLoopCounter], tags[innerLoopCounter-clfWindowSize:innerLoopCounter])
        sentIndecies= sentIndecies | set(range(innerLoopCounter-clfWindowSize,innerLoopCounter))
    
    time+=1
    
print syncs
print len(syncs)

dic={}
dic['Nodes num'] = k
dic['Window size for positive class'] = allDataP.shape[1]
dic['Window size for negative class'] = allDataQ.shape[1]
dic['Dimension'] = d
dic['Threshold'] = T
dic['Number of syncs'] = len(syncs)
dic['violationThreshold'] = violationThreshold
dic['clfWindowSize'] = clfWindowSize
dic['SentInstances'] = len(sentIndecies)
conceptsDrifts = [chunkSize*i/k for i in range(1, numOfchunks)]

plt.title(str(dic))
plt.plot(params, accuracies, label='Accuracy '+str(hits/testDataLen))
plt.plot(params, adaAccuracies, label='Adaptive accuracy '+str(adaHits/testDataLen))
plt.plot(params, oracleAccuracies, label='Oracle accuracy '+str(oracleHits/testDataLen))
plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
            label='Concepts Drifts', marker='x', s=100)
for sync in syncs:
    plt.axvline(sync, color='r')
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Accuracy')

plt.figure()

#plt.plot(params,cosines, label='True cosine simillarity')
plt.plot(params,leftValues, label='DLDA Bound')
plt.plot(params,reals, label='norm(w-w0)')
plt.plot(params,R0s, label='R0')

plt.scatter(syncs, np.ones_like(syncs), c='b', label='Syncs')
#[300,600,900,1200]
#plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
#            label='Concepts Drifts', marker='x', s=100)
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Error')
plt.title(str(dic))

plt.show() 
