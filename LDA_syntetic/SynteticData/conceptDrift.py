import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
mu_p_0 = np.array([0.0, 0.0])
R=2.0
mu_q_0 = np.array([R,0.0])
var=1
cov_p_0 = np.array([[var,0],\
                    [0,var]])
cov_q_0 = np.array([[var,0],\
                    [0,var]])

d=len(mu_p_0)
k=1
#L=200
dic={}
#beforSyncLength = L
timeLength=300#2*L
T=0.9


params = np.linspace(0, np.pi/6, 50)
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
#syncsNum = np.zeros((len(params),))
repeatTime = 1
def foo():
    for L in [10000]:
        violationCounters= np.zeros((len(params),))
        cosines = np.zeros((len(params),))
        violationsNum = np.zeros((len(params),))
        for t,theta in enumerate(params):
            tmp = R*np.cos(theta)
            mu_q[0] = tmp
            mu_q[1] = R*np.sin(theta)
            mu_p = mu_p_0
            for r in range(repeatTime):
                allDataP = np.zeros((k,L,d))
                allDataQ = np.zeros((k,L,d))
                
                allDataPbefore = np.zeros((k,L,d))
                allDataQbefore = np.zeros((k,L,d))
                nodeData=[]
                for i in range(k):
                    Xp_0 = np.random.multivariate_normal(mu_p_0, cov_p, L)
                    Xq_0 = np.random.multivariate_normal(mu_q_0, cov_q, L)
                    allDataPbefore[i] = Xp_0
                    allDataQbefore[i] = Xq_0
                    referenceParams = getXYS(Xp_0, Xq_0)
                    #nodeData.append((Xp_0,Xq_0,referenceParams))
                    Xp = np.random.multivariate_normal(mu_p, cov_p, L)
                    Xq = np.random.multivariate_normal(mu_q, cov_q, L)
                    allDataP[i] = Xp
                    allDataQ[i] = Xq
                    nodeData.append((Xp,Xq,referenceParams))
                S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataPbefore, allDataQbefore)
                for time in range(timeLength):
                    violationCounter = 0
                    for i in range(k):
                        dataP, dataQ, referenceParams = nodeData[i]
                        dataP, dataQ = (dataP[1:], dataQ[1:]) 
                        newP = np.random.multivariate_normal(mu_p, cov_p,1)
                        newQ = np.random.multivariate_normal(mu_q, cov_q,1)
                        dataP = np.concatenate((dataP, newP))
                        dataQ = np.concatenate((dataQ, newQ))
                        nodeData[i] = (dataP, dataQ, referenceParams)
                        allDataP[i] = np.concatenate((allDataP[i][1:],newP))
                        allDataQ[i] = np.concatenate((allDataQ[i][1:],newQ))
            
                        x0_i, y0_i, S0_i = referenceParams
                        isInSafeZone = checkLocalConstraint(S0_i, x0_i, y0_i, w0_norm, \
                            B0_inverted, dataP, dataQ, T)
                        if not isInSafeZone:
                            violationsNum[t]+=1.0
                            violationCounter += 1
                    
                    S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
                    c = cosineSimilarity(w0, w)   
                    if c < T:
                        cosines[t]+=1.0
                    dic['Sync'] = 'No'
                    """
                    if violationCounter > k/2:#Lets sync
                        syncsNum[t] +=1.0 
                        for i in range(k):
                            dataP, dataQ, referenceParams = nodeData[i]
                            referenceParams = getXYS(dataP, dataQ)
                            nodeData[i] = (dataP, dataQ, referenceParams)
                        S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ) 
                        dic['Sync'] = 'Yes'    
                    """    
        violationsNum /= (repeatTime*timeLength*k)
        cosines /= (repeatTime*timeLength*k)
        plt.plot(params, violationsNum, label='Node violations, L='+str(L), marker='<' )
        plt.plot(params, cosines, label='True violations, L='+str(L), marker='>' )
        #syncsNum /= (repeatTime*timeLength)
dic['k'] = k
#dic['L'] = L
dic['repeatTime'] = repeatTime
dic['timeLength'] = timeLength
dic['T(In radians)'] = np.arccos(T)
dic['var'] = var
#plt.plot(params, syncsNum, label='Syncs' )
plt.title(str(dic))
plt.xlabel('Theta (in radians)')
plt.ylabel('Probability for violation')
#plt.legend().draggable()
plt.ylim(-0.1, 1.1)

import cProfile
cProfile.run('foo()')
plt.show()   
    