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
k=1
L=80
beforSyncLength = L
timeLength=L/2
dataLength = beforSyncLength + timeLength*k

repeatTime = 160
#Ts = 1 -np.logspace(0,-5,20)
Ts = np.concatenate((np.linspace(0, 0.9, 20),np.linspace(0.9, 1, 20)))
violationsNum = np.zeros(Ts.shape)
cosines = np.zeros(Ts.shape)
for t,T in enumerate(Ts): 
    for r in range(repeatTime):
        Xp =  np.random.multivariate_normal(mu_p_0.T[0], cov_p_0, dataLength)
        Xq =  np.random.multivariate_normal(mu_q_0.T[0], cov_q_0, dataLength)
        Xp_0 =  Xp[:beforSyncLength]
        Xq_0 =  Xq[:beforSyncLength]
        Xp =  Xp[beforSyncLength:]
        Xq =  Xq[beforSyncLength:]
        allDataP = np.repeat([Xp_0], [k], axis=0)
        allDataQ = np.repeat([Xq_0], [k], axis=0)
        nodeData=[]
        for i in range(k):
            nodeData.append((Xp_0,Xq_0))
        time=0
        S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams(Xp_0, Xq_0)
        while time < timeLength:
            for i in range(k):
                dataP, dataQ = nodeData[i]
                #S_0, x_0, y_0, w0, w_0_norm, B_0_inverted \
                #    = calcWindowParams(dataP, dataQ)
                
                index = time*k+i
                dataP, dataQ = (dataP[1:], dataQ[1:]) 
                newP = np.reshape(Xp[index], (1,2))
                newQ = np.reshape(Xq[index], (1,2))
                dataP = np.concatenate((dataP, newP))
                dataQ = np.concatenate((dataQ, newQ))
                
                nodeData[i] = (dataP, dataQ)
                isInSafeZone = checkLocalConstraint(S0, x0, y0, w0_norm, \
                    B0_inverted, dataP, dataQ, T)
                if not isInSafeZone:
                    violationsNum[t]+=1.0
                
                allDataP[i] = np.concatenate((allDataP[i][1:],newP))
                allDataQ[i] = np.concatenate((allDataQ[i][1:],newQ))
                allDataPStacked = allDataP.reshape((k*L,d))
                allDataQStacked = allDataQ.reshape((k*L,d))
                S, x, y, w, w_norm, B_inverted = \
                    calcWindowParams(allDataPStacked, allDataQStacked) 
                c = cosineSimilarity(w0, w)   
                if c < T:
                    cosines[t]+=1.0
            time+=1
dic={}
dic['k'] = k
dic['L'] = L
dic['repeatTime'] = repeatTime
dic['timeLength'] = timeLength
violationsNum = violationsNum/repeatTime/timeLength/k
cosines = cosines/repeatTime/timeLength/k
plt.plot(Ts, violationsNum, label='Node violations' )
plt.plot(Ts, cosines, label='True violations' )
plt.title(str(dic))
plt.xlabel('T')
plt.ylabel('Probability for violation')
plt.legend().draggable()
plt.show()
