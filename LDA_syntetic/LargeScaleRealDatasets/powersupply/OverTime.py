import numpy as np
from utils import *
from sklearn.lda import LDA
from sklearn.decomposition import PCA
import copy
import itertools

k=50
#L=400
T = 0.5
numberOfdaysInWindow=500
clfWindowSize = numberOfdaysInWindow*24/k
trajWindow= 2000#clfWindowSize*k
initLen = clfWindowSize*k
violationThreshold = k-5
d=2
alpha=1

f = open('C:/Users/ran/Downloads/powersupply.arff')
X=[]
Xp=[]
Xq=[]
tags=[]
for j,line in enumerate(f):
    if '\n' in line:
        line=line[:-1]
        splited=line.split(',')
    hour = int(splited[2])
    tag=(hour > 7 and hour < 20)
    #tag=int(splited[-1])
    x = [float(v) for v in splited[:2]]
    X.append(np.array(x))
    tags.append(tag)
    
    if tag:
        Xp.append(x)
    else:
        Xq.append(x)
print len(tags)
allDataP = []
allDataQ = []
allDataPFlat = []
allDataQFlat = []
for i in range(k):
    allDataP.append([])
    allDataQ.append([])
pI=0
qI=0
for time in range(initLen/k):
    violationCounter = 0.
    for i in range(k): 
        index=time*k+i
        newPoint =  X[index]
        tag = tags[index]
        if tag:
            allDataP[i].append(newPoint)
            allDataPFlat.append(newPoint)
            pI+=1
        else:
            allDataQ[i].append(newPoint)
            allDataQFlat.append(newPoint)
            qI+=1
references=[]
for i in range(k): 
    referenceParams = getXYS(allDataP[i], allDataQ[i])
    references.append(referenceParams)
allDataP = np.reshape(np.array(allDataP),(k,pI/k,d))
allDataQ = np.reshape(np.array(allDataQ),(k,qI/k,d))
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
Psize = len(allDataP[0])
print len(allDataP[0])
Qsize = len(allDataQ[0])
print len(allDataQ[0])
dataLength = len(X) 
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
time=initLen/k
TVcounter = np.zeros(k+1)
NTVcounter = np.zeros(k+1)
params=[]
p0s=[]
p1s=[]
q0s=[]
q1s=[]
#pI,qI = trajWindow, trajWindow
trajParams=[]#
Xp, Xq = np.array(Xp), np.array(Xq)
for i in range(initLen,dataLength):
    if pI>len(Xp) or qI>len(Xq):
        break
    tag = tags[i]
    if tag:
        pI+=1
    else:
        qI+=1
    trajParams.append(pI+qI)
    #S, x, y, w, w_norm, B = \
    #    calcWindowParams(Xp[pI-min(trajWindow,pI):pI], \
    #                     Xq[qI-min(trajWindow,qI):qI])
    #w =w/norm(w)
    #if w.item(0) < 0:
    #    w=w*(-1)
    #p0s.append(w.item(0))
    #p1s.append(w.item(1))
    p0s.append(np.mean(Xp[pI-min(trajWindow,pI):pI,0]))
    p1s.append(np.mean(Xp[pI-min(trajWindow,pI):pI,1]))
    q0s.append(np.mean(Xq[qI-min(trajWindow,qI):qI,0]))
    q1s.append(np.mean(Xq[qI-min(trajWindow,qI):qI,1]))
trajParams=np.array(trajParams)
innerLoopCounter = initLen
while innerLoopCounter < dataLength-k+1:    
    R0 = getR0(w0_norm, T)
    
    violationCounter = 0.
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
        
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
    
    params.append(time)
    leftValues.append(violationCounter/k)
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
        
        leftValues.append(0)
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
        real = norm(w-w0)
        real /= R0
        reals.append(real)
        params.append(time)
    #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
    #c = cosineSimilarity(w0, w) 
    #cosines.append(c)
    
    time+=1
    
print syncs
print len(syncs)

#params = range(initLen/k, time)
dic={}
dic['trajWindow']=trajWindow
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
"""
plt.title(str(dic))
plt.plot(params, accuracies, label='Accuracy '+str(hits/testDataLen))
plt.plot(params, adaAccuracies, label='Adaptive accuracy '+str(adaHits/testDataLen))

for sync in syncs:
    plt.axvline(sync, color='r')
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Accuracy')
"""

"""
plt.figure()
hist=[]
for i in range(k+1):
    hist.append(TVcounter[i]/(TVcounter[i]+NTVcounter[i]))
plt.scatter(range(k+1),hist)
plt.title(str(dic))
plt.xlabel('Number of locally violated nodes')
plt.ylabel('Probability for true violation')
"""

"""
plt.figure()
plt.plot(trajParams,p0s, label='p[0]')
plt.plot(trajParams,p1s, label='p[1]')
plt.plot(trajParams,q0s, label='q[0]')
plt.plot(trajParams,q1s, label='q[1]')
for sync in syncs:
    plt.axvline(sync*k, color='r')
plt.legend().draggable()
plt.xlabel('Instance Number')
plt.ylabel('Position')
plt.title(str(dic))
"""

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