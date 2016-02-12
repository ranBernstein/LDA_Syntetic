import numpy as np
from utils import *
from sklearn.lda import LDA
from sklearn.decomposition import PCA
import copy
import itertools
k=2
#L=100
T = 0.5

#numOfchunks=5
#chunkSize = 300
clfWindowSize = 1000
initLen = clfWindowSize
violationThreshold = 0#k-1
d=3
alpha=1#1.0/d
#firstClass = [1,3,4]
classes = [1,2]
def readBatch(batchNum):
    dataFile = open('batch'+str(batchNum)+'.dat', 'r')
    X=[]
    tags=[]
    for line in dataFile:
        line = line.strip()
        splited= line.split()
        x = [float(v.split(':')[1]) for v in splited[1:]]
        tag= int(splited[0])
        #tag = (tag in firstClass)
        if tag in classes:
            tag = (tag>classes[0])
            X.append(x)
            tags.append(tag)
    #return np.array(X), np.array(tags)
    return X, tags

def concatenateBatches():
    X=[]
    tags=[]
    for batchNum in range(1,11):
        currX, currTags = readBatch(batchNum)
        X = X + currX
        tags = tags + currTags
    return np.array(X), np.array(tags)

def readBatchWithPCA(batchNum, pca):
    X, tags = readBatch(batchNum)
    if pca:
        X = pca.transform(X)
    return X, tags
 
X, tags =  concatenateBatches()  
dataLength = len(X) 
print X.shape
print tags.shape


"""
trainX,trainTags = readBatch(1)
pca = PCA(n_components=d)
#trainX = pca.fit_transform(trainX,trainTags)
#print(pca.explained_variance_ratio_) 

clf = LDA()
clf.fit(trainX, trainTags)
#combs = list(itertools.combinations((1, 2, 3, 4, 5, 6), 3))
for batchNum in range(1,11):
    trainX,trainTags = readBatchWithPCA(batchNum, False)
    clf.fit(trainX, trainTags)
    params = range(batchNum,11)
    scores=[]
    for batchNum2 in params: 
        testX, testTags = readBatchWithPCA(batchNum2, False)
        scores.append(clf.score(testX, testTags)) 

    plt.plot(params, scores)
plt.title(str(firstClass))
plt.show()
"""

pca = PCA(n_components=d)
X = pca.fit_transform(X,tags)
print'pca.explained_variance_ratio_', pca.explained_variance_ratio_
print np.sum(pca.explained_variance_ratio_), 'np.sum(pca.explained_variance_ratio_)'
#L=100
allDataP = []
allDataQ = []
allDataPFlat = []
allDataQFlat = []
for i in range(k):
    allDataP.append([])
    allDataQ.append([])
for time in range(initLen):
    violationCounter = 0
    for i in range(k): 
        newPoint =  X[time]
        tag = tags[time]
        if tag:
            allDataP[i].append(newPoint)
            allDataPFlat.append(newPoint)
        else:
            allDataQ[i].append(newPoint)
            allDataQFlat.append(newPoint)
references=[]
for i in range(k): 
    referenceParams = getXYS(allDataP[i], allDataQ[i])
    references.append(referenceParams)
allDataP = np.array(allDataP)
allDataQ = np.array(allDataQ)
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
Psize = len(allDataP[0])
print len(allDataP[0])
Qsize = len(allDataQ[0])
print len(allDataQ[0])


clf = LDA()
clf.fit(X[:initLen], tags[:initLen])

adaClf = LDA()
adaClf.fit(X[:initLen], tags[:initLen])

lastRes = {}
pred = clf.predict(X[initLen-clfWindowSize:initLen])
real =  tags[initLen-clfWindowSize:initLen]
lastRes = np.logical_not(np.logical_xor(pred, real))
adaLastRes = copy.copy(lastRes)
#dataLength = numOfchunks*chunkSize
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
testDataLen = dataLength - initLen
sentIndecies = set()
innerLoopCounter = initLen
time=initLen/k
TVcounter = np.zeros(k+1)
NTVcounter = np.zeros(k+1)
params=[]
while innerLoopCounter < dataLength-k+1:    
    R0 = getR0(w0_norm, T)
    params.append(time)
    violationCounter = 0
    leftValue = []
    for i in range(k): 
        newPoint =  X[innerLoopCounter].reshape((1,d))
        tag = tags[innerLoopCounter]
        if tag:
            allDataP[i] = np.concatenate((allDataP[i][1:], newPoint))
            allDataPFlat.append(X[innerLoopCounter])
        else:
            allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint))
            allDataQFlat.append(X[innerLoopCounter])

        x0_i, y0_i, S0_i = references[i]
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
        res = clf.predict([X[innerLoopCounter]])
        lastRes[windowIndex] = (res==tag)
        hits+=(res==tag)
        
        adaRes = adaClf.predict([X[innerLoopCounter]])
        adaLastRes[windowIndex] = (adaRes==tag)
        adaHits+=(adaRes==tag)
        
        innerLoopCounter+=1
    
    accuracy = float(np.sum(lastRes))/clfWindowSize
    accuracies.append(accuracy)
    adaAccuracy = float(np.sum(adaLastRes))/clfWindowSize
    adaAccuracies.append(adaAccuracy)
    
    leftValue = np.max(leftValue)
    leftValue /= R0
    leftValues.append(leftValue)
    
    S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
    real = norm(w-w0)
    real /= R0
    reals.append(real)
    if real > 1 :
        TVcounter[violationCounter]+=1
    else:
        NTVcounter[violationCounter]+=1

        
    
    #if real > 2:
    if violationCounter >violationThreshold:
        syncs.append(time)
        for i in range(k):
            referenceParams = getXYS(allDataP[i], allDataQ[i])
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
        u0 = x0-y0        
        newData = np.concatenate((allDataPFlat[-Psize:], allDataQFlat[-Qsize:]))
        newTags = np.concatenate((np.ones(Psize), np.zeros(Qsize)))
        adaClf.fit(newData, newTags)
        #adaClf.fit(X[innerLoopCounter-clfWindowSize:innerLoopCounter], tags[innerLoopCounter-clfWindowSize:innerLoopCounter])
        sentIndecies= sentIndecies | set(range(innerLoopCounter-clfWindowSize,innerLoopCounter))
    #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
    #c = cosineSimilarity(w0, w) 
    #cosines.append(c)
    
    time+=1
    
print syncs
print len(syncs)

params = range(initLen/k, time)
dic={}
dic['Nodes num'] = k
#dic['Window size for positive class'] = allDataP.shape[1]
#dic['Window size for negative class'] = allDataQ.shape[1]
dic['Dimension'] = d
dic['Threshold'] = T
#dic['Number of syncs'] = len(syncs)
#dic['violationThreshold'] = violationThreshold
dic['clfWindowSize'] = clfWindowSize
#dic['SentInstances'] = len(sentIndecies)
dic['Rounds/syncs_Ratio'] = len(params)/len(syncs)

plt.title(str(dic))
plt.plot(params, accuracies, label='Accuracy '+str(hits/testDataLen))
plt.plot(params, adaAccuracies, label='Adaptive accuracy '+str(adaHits/testDataLen))

for sync in syncs:
    plt.axvline(sync, color='r')
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Accuracy')

plt.figure()
hist=[]
for i in range(k+1):
    hist.append(TVcounter[i]/(TVcounter[i]+NTVcounter[i]))
plt.scatter(range(k+1),hist)
plt.title(str(dic))
plt.xlabel('Number of locally violated nodes')
plt.ylabel('Probability for true violation')
plt.figure()

#plt.plot(params,cosines, label='True cosine simillarity')
plt.plot(params,leftValues, label='DLDA Bound')
plt.plot(params,reals, label='norm(w-w0)')
#plt.plot(params,R0s, label='R0')

plt.scatter(syncs, np.ones_like(syncs), c='b', label='Syncs')
#[300,600,900,1200]
#plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
#            label='Concepts Drifts', marker='x', s=100)
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Error')
plt.title(str(dic))


plt.show() 
