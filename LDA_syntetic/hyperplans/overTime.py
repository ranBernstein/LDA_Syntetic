from utils import *
from sklearn.decomposition import PCA
import copy

numOfchunks=1
chunkSize = 10000
clfWindowSize = 1000
initLen = clfWindowSize
d=10
k=10
T = 0.93

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
    
    return X, tags
    
    

X=[]
tags=[]
for DataSetNum in range(1,10):
    currX, currTags = getDataSet(DataSetNum)
    if len(X) ==0:
        X = currX
        tags = currTags
    else:
        X = X + currX
        tags = tags + currTags
#print len(X)

X = np.array(X)
tags = np.array(tags)
pca = PCA(n_components=d)
X = pca.fit_transform(X,tags)
print'pca.explained_variance_ratio_', pca.explained_variance_ratio_
#L=100
allDataP = []
allDataQ = []
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
print len(allDataP[0])
print len(allDataQ[0])


from sklearn.lda import LDA
clf = LDA()
clf.fit(X[:initLen], tags[:initLen])

adaClf = LDA()
adaClf.fit(X[:initLen], tags[:initLen])

lastRes = {}
pred = clf.predict(X[initLen-clfWindowSize:initLen])
real =  tags[initLen-clfWindowSize:initLen]
lastRes = np.logical_not(np.logical_xor(pred, real))
adaLastRes = copy.copy(lastRes)
dataLength = numOfchunks*chunkSize
syncs = []
cosines = []
timeLength = (dataLength-initLen)/k
params = range(timeLength)
accuracies=[]
adaAccuracies=[]
innerLoopCounter = initLen
time=0
while innerLoopCounter < dataLength:
    R0 = getR0(w0_norm, T)
    violationCounter = 0
    #accuracy = 0.0
    for i in range(k): 
        newPoint =  X[innerLoopCounter].reshape((1,d))
        tag = tags[innerLoopCounter]
        if tag:
            allDataP[i] = np.concatenate((allDataP[i][1:], newPoint))
        else:
            allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint))
        globalParams=w0, B0, u0, w
        currentData=allDataP[i], allDataQ[i]
        isInSafeZone = checkLocalConstraint(references[i],  
                            globalParams, currentData, R0, alpha=1)
        if not isInSafeZone:
            violationCounter += 1
        windowIndex = innerLoopCounter%clfWindowSize
        res = clf.predict([X[innerLoopCounter]])
        lastRes[windowIndex] = (res==tag)
        
        adaRes = adaClf.predict([X[innerLoopCounter]])
        adaLastRes[windowIndex] = (adaRes==tag)
        #nodeAccuracy = float(np.sum(lastRes))/clfWindowSize
        #accuracy+=nodeAccuracy
        innerLoopCounter+=1
    accuracy = float(np.sum(lastRes))/clfWindowSize
    accuracies.append(accuracy)
    
    adaAccuracy = float(np.sum(adaLastRes))/clfWindowSize
    adaAccuracies.append(adaAccuracy)
    
    if violationCounter > 0:
        syncs.append(time)
        for i in range(k):
            referenceParams = getXYS(allDataP[i], allDataQ[i])
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
        u0 = x0-y0 
        adaClf.fit(X[innerLoopCounter-clfWindowSize:innerLoopCounter], tags[innerLoopCounter-clfWindowSize:innerLoopCounter])
    S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
    c = cosineSimilarity(w0, w) 
    cosines.append(c)
    time+=1
print syncs
print len(syncs)

#plt.plot(params,cosines, label='True cosine simillarity')
#plt.scatter(syncs, np.ones_like(syncs), c='b', label='Syncs')
plt.plot(params, accuracies, label='Accuracy')
plt.plot(params, adaAccuracies, label='Adaptive accuracy')
#conceptsDrifts = range(chunkSize, length+1, chunkSize)
for sync in syncs:
    plt.axvline(sync, color='r')
#plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
#            label='Concepts Drifts', marker='x', s=100)
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Accuracy')
dic={}
dic['Nodes num'] = k
dic['Window size for positive class'] = allDataP.shape[1]
dic['Window size for negetibe class'] = allDataQ.shape[1]
dic['Dimension'] = d
dic['Threshold'] = T
dic['Number of syncs'] = len(syncs)
plt.title(str(dic))
plt.show()