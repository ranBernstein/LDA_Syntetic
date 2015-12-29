from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import copy
from utils import *
import pylab as P
k=1
d=2
T = 0.5
alpha = 1
clfWindowSize = 150
initLen = clfWindowSize
violationThreshold = k-1

threshs = []
ds = range(1,20)
for d in ds:
    wastes = []
    digits = datasets.load_digits()
    from sklearn.preprocessing import scale
    data = scale(digits.data)
    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target
    X = []
    y = []
    binaryLabels = []
    pca = PCA(n_components=d)
    data = pca.fit_transform(data, labels)
    dataPerDigit=[]
    allP=[]
    allQ=[]
    for i in range(6):
        indecies = np.where(labels==i)[0]
        currDigit = data[indecies].tolist()
        dataPerDigit.append(currDigit)
        X = X + currDigit
        if i%2==0:
            allP = allP + currDigit
            currLabels = np.zeros(len(currDigit)).tolist()
        else:
            allQ = allQ + currDigit
            currLabels = np.ones(len(currDigit)).tolist()
        y = y + currLabels
            
    allDataP = []
    allDataQ = []
    allDataPFlat = []
    allDataQFlat = []
    for i in range(k):
        allDataP.append([])
        allDataQ.append([])
    inner=0
    for time in range(initLen/k/2):
        for i in range(k): 
            newP = allP[inner]
            allDataP[i].append(newP)
            allDataPFlat.append(newP)
            
            newQ = allQ[inner]
            allDataQ[i].append(newQ)
            allDataQFlat.append(newQ)
            inner+=1
    references=[]
    for i in range(k): 
        referenceParams = getXYS(allDataP[i], allDataQ[i])
        references.append(referenceParams)  
    allDataP = np.array(allDataP)
    allDataQ = np.array(allDataQ)
    S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
    u0 = x0-y0    
    dataLength = len(X)
    timeLengh = (dataLength - initLen)/k 
    
    clf = LDA()
    Psize, Qsize = clfWindowSize/2, clfWindowSize/2
    trainX = allDataPFlat[:Psize]+allDataQFlat[:Qsize]
    trainY = np.concatenate((np.zeros(Psize), np.ones(Qsize)))
    clf.fit(trainX, trainY)
    
    adaClf = LDA()
    adaClf.fit(trainX, trainY)
    
    lastRes = {}
    pred = clf.predict(X[initLen-clfWindowSize:initLen])
    real =  y[initLen-clfWindowSize:initLen]
    lastRes = np.logical_not(np.logical_xor(pred, real))
    adaLastRes = copy.copy(lastRes)
    
    sentIndecies = set()
    innerLoopCounter = initLen
    time=initLen/k
    
    leftValues = []
    reals =[]
    R0s =[]
    
    hits=0.0
    adaHits=0.0
    oracleHits=0.0
    
    accuracies=[]
    adaAccuracies=[]
    syncs=[]
    allP, allQ = np.array(allP), np.array(allQ)
    waistIntterput =0
    while innerLoopCounter < 2*min(len(allP), len(allQ))-k:    
        R0 = getR0(w0_norm, T)
        
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
        real = norm(w-w0)
        reals.append(real)
        leftValue = 0
        for i in range(k): 
            #x0_i, y0_i, S0_i = references[i]
            #localParams=S0_i, x0_i, y0_i
            globalParams=w0, B0, u0
            currentData=allDataP[i], allDataQ[i]
            currLeftValue, waste = getLeftSide(references[i],  globalParams, 
                                               currentData, R0, alpha)
            leftValue+=currLeftValue
        leftValue /= k
        leftValues.append(leftValue)
        R0s.append(R0)
        violationCounter = 0
        for i in range(k): 
            tag = innerLoopCounter%2
            if not tag:
                newPoint =  allP[innerLoopCounter/2]
                allDataP[i] = np.concatenate((allDataP[i][1:], newPoint.reshape((1,d))))
                allDataPFlat.append(newPoint)
            else:
                newPoint =  allQ[innerLoopCounter/2]
                allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint.reshape((1,d))))
                allDataQFlat.append(newPoint)
    
            globalParams=w0, B0, u0
            currentData=allDataP[i], allDataQ[i]
            currLeftValue, waste = getLeftSide(references[i],  globalParams, 
                                               currentData, R0, alpha)
            wastes.append(waste)
            if currLeftValue > R0 and currLeftValue-waste[-1]<=R0:
                #print currLeftValue, R0, waste[-1]
                waistIntterput+=1
            isInSafeZone = (currLeftValue <= R0)
            #isInSafeZone = checkLocalConstraint(S0_i, x0_i, y0_i, w0, B0, w,\
            #    allDataP[i], allDataQ[i], T)
            if not isInSafeZone:
                violationCounter += 1
            
            windowIndex = innerLoopCounter%clfWindowSize 
            res = clf.predict(newPoint)[0]
            lastRes[windowIndex] = (res==tag)
            hits+=(res==tag)
            
            adaRes = adaClf.predict(newPoint)[0]
            adaLastRes[windowIndex] = (adaRes==tag)
            adaHits+=(adaRes==tag)
            
            innerLoopCounter+=1
        
        accuracy = float(np.sum(lastRes))/clfWindowSize
        accuracies.append(accuracy)
        adaAccuracy = float(np.sum(adaLastRes))/clfWindowSize
        adaAccuracies.append(adaAccuracy)
        
        
        if violationCounter >violationThreshold:
            syncs.append(time)
            for i in range(k):
                referenceParams = getXYS(allDataP[i], allDataQ[i])
                references[i] = referenceParams
            S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
            u0 = x0-y0 
            newData = np.concatenate((allDataPFlat[-Psize:], allDataQFlat[-Qsize:]))
            newTags = np.concatenate((np.zeros(Psize), np.ones(Qsize)))
            adaClf.fit(newData, newTags)
            #adaClf.fit(X[innerLoopCounter-clfWindowSize:innerLoopCounter], tags[innerLoopCounter-clfWindowSize:innerLoopCounter])
            sentIndecies= sentIndecies | set(range(innerLoopCounter-clfWindowSize,innerLoopCounter))
        #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        #c = cosineSimilarity(w0, w) 
        #cosines.append(c)
        
        time+=1
        
    
    params = range(initLen/k, time)
    dic={}
    dic['Nodes num'] = k
    dic['Window size for positive class'] = allDataP.shape[1]
    dic['Window size for negative class'] = allDataQ.shape[1]
    dic['Dimension'] = d
    dic['Threshold'] = T
    dic['Number of syncs'] = len(syncs)
    dic['violationThreshold'] = violationThreshold
    dic['alpha'] = alpha
    #dic['clfWindowSize'] = clfWindowSize
    #dic['SentInstances'] = len(sentIndecies)
    dic['Rounds/syncs_Ratio'] = len(params)/len(syncs)
    """
    plt.title(str(dic))
    testDataLen = dataLength - initLen
    plt.plot(params, accuracies, label='Accuracy '+str(hits/testDataLen))
    plt.plot(params, adaAccuracies, label='Adaptive accuracy '+str(adaHits/testDataLen))
    
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
    plt.figure()
    
    from scipy.stats import lognorm
    wastes = np.array(wastes).reshape((len(wastes),len(wastes[0])))
    wastes=wastes[:,3]
    wastes.sort()
    weights = np.ones_like(wastes)/float(len(wastes))
    shape, loc, scale = lognorm.fit(wastes)
    
    P.hist(wastes,50, normed=0, weights=weights, label='Data')
           #,label = ['1','2','3', '4', '5', 'All together'])
    dic['mu'] = loc
    dic['sigma']=shape
    dic['scale']=scale
    print  shape, loc, scale
    plt.plot(wastes, lognorm.pdf(wastes, shape, loc, scale),'r-', lw=1, alpha=1, label='expon pdf')
    plt.xlabel('Waste/R0')
    plt.ylabel('Distribution')
    plt.legend().draggable()
    
    plt.title(str(dic))
    
    plt.figure()
    plt.plot(wastes, lognorm.cdf(wastes, shape, loc, scale))
    """
    threshs.append(lognorm.ppf([0.05], shape, loc, scale))
plt.plot(ds, threshs)
plt.xlabel('Dimension')
plt.ylabel('lognorm.ppf([0.05])')
plt.title('Analysis of the distribution of \
    normOperator(B0_inverted*Delta)*w0_norm/norm(B0_inverted*Delta*w0)')
"""
for i in range(wastes.shape[1]):
    print np.mean(wastes[:,i])
print waistIntterput
"""
#print wastes
#print (dataLength-initLen)
plt.show() 