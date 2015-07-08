import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *

mu_p_0 = np.array([[2],[2]])
mu_q_0 = np.array([[-2],[-2]])
cov_p_0 = np.array([[1,1],\
                    [1,2]])
cov_q_0 = np.array([[2,-1],\
                    [-1,1]])
d=len(mu_p_0)
k=40
L=200
T=0.9
dic={}

beforSyncLength = L
timeLength=50*L
dataLength = beforSyncLength + timeLength*k

Ts = np.concatenate((np.linspace(0, 0.9, 20),np.linspace(0.9, 1, 20)))
violationsNum = np.zeros(Ts.shape)
cosines = np.zeros(Ts.shape)
syncsNum = np.zeros(Ts.shape)

allDataP = np.zeros((k,L,d))
allDataQ = np.zeros((k,L,d))
nodeData=[]
for i in range(k):
    Xp_0 = np.random.multivariate_normal(mu_p_0.T[0], cov_p_0, L)
    Xq_0 = np.random.multivariate_normal(mu_q_0.T[0], cov_q_0, L)
    allDataP[i] = Xp_0
    allDataQ[i] = Xq_0
    referenceParams = getXYS(Xp_0, Xq_0)
    nodeData.append((Xp_0,Xq_0,referenceParams))

S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
violationCounters= np.zeros(timeLength,)
cosines = np.zeros(timeLength,)
params = range(timeLength)
for time in params:
    violationCounter = 0
    for i in range(k):
        dataP, dataQ, referenceParams = nodeData[i]
        dataP, dataQ = (dataP[1:], dataQ[1:]) 
        newP = np.random.multivariate_normal(mu_p_0.T[0], cov_p_0,1)
        newQ = np.random.multivariate_normal(mu_q_0.T[0], cov_q_0,1)
        dataP = np.concatenate((dataP, newP))
        dataQ = np.concatenate((dataQ, newQ))
        nodeData[i] = (dataP, dataQ, referenceParams)
        
        allDataP[i] = np.concatenate((allDataP[i][1:],newP))
        allDataQ[i] = np.concatenate((allDataQ[i][1:],newQ))
        
        x0_i, y0_i, S0_i = referenceParams
        isInSafeZone = checkLocalConstraint(S0_i, x0_i, y0_i, w0_norm, \
            B0_inverted, dataP, dataQ, T)
        if not isInSafeZone:
            #violationsNum[t]+=1.0
            violationCounter += 1
    violationCounters[time] = violationCounter
    S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
    c = cosineSimilarity(w0, w)
    cosines[time] = c
    
    if violationCounter > k/2:#Lets sync
        #syncsNum[t] +=1.0 
        for i in range(k):
            dataP, dataQ, referenceParams = nodeData[i]
            referenceParams = getXYS(dataP, dataQ)
            nodeData[i] = (dataP, dataQ, referenceParams)
        S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ) 
        dic['Sync'] = 'Yes'
    time+=1

dic['k'] = k
dic['L'] = L
dic['T'] = T
dic['timeLength'] = timeLength
violationCounters /= k
cosines -= 1
cosines *= 500
cosines += 1
plt.plot(params, violationCounters, label='Probability for violation in a node')
plt.plot(params, cosines, label='500*(1-CosinesSimilarities) + 1')

plt.title(str(dic))
plt.xlabel('Time')
#plt.ylabel('Probability for violation in a node')
plt.legend().draggable()
plt.show()
