import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
dic={}
k=60
dic['k']=k

"""
x = np.arange(-100, 100)
p = chernof(x)
plt.plot(x, p)
plt.show()
"""

#violationThreshold=100
#dic['violationThreshold'] = violationThreshold

alpha=1
#dic['alpha']=alpha
d=2
dic['d']=d
L=50
dic['L']=L
T=0.9
dic['T']=T
periodSize = 2*L
dic['periodSize']=periodSize
periodNum = 2
#dic['periodNum']=periodNum
R=2.0
dic['R']=R
mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
mu_q_0[0] = R
var=0.01
dic['var']=var
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = copy.deepcopy(cov_p_0)
repeatTime=10

falseNegatives=[]
falsePositives=[]
truePositives=[]
params=range(k/3-1,k,k/6)
for violationThreshold in params:
    print violationThreshold
    for _ in range(repeatTime):
        allDataP, allDataQ, references  = \
            initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
        
        mu_p = copy.copy(mu_p_0)
        mu_q = copy.copy(mu_q_0)
        cov_p = copy.copy(cov_p_0)
        cov_q = copy.copy(cov_q_0)
        priorData=[]
        for period in range(periodNum):
            theta = 2*np.pi/periodNum*period
            tmp = R*np.cos(theta)
            mu_q[0] = tmp
            mu_q[1] = R*np.sin(theta)
            mu_p = mu_p_0
            Xp = np.random.multivariate_normal(mu_p, cov_p, k*periodSize/2)
            Xq = np.random.multivariate_normal(mu_q, cov_q,  k*periodSize/2)
            priorData.append((Xp,Xq))
            
        #rounds=1000
        localFalseNegatives=[]
        localFalsePositives=[]
        localTruePositives=[]
    
        #params = []#range(rounds)
        
        S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
        u0 = x0-y0
        time = 0
        innerLoopCounter=0
        leftValues=[]
        magnitudeDifs =[]
        reals=[]
        syncs=[]
        conceptsDrifts=[]
        falseNegative=0.
        falsePositive=0.
        truePositive=0.
        trueViolationCounter=0.
        totalTrueViolations=0.
        totalFalseViolations=0
        for period in range(periodNum):
            Xp,Xq = priorData[period]
            conceptsDrifts.append(time)
            for timeInPeriod in range(periodSize):
                #params.append(time)
                R0 = getR0(w0_norm, T)
                violationCounter = 0.0
                leftValue = []
                for i in range(k):
                    index = k*timeInPeriod + i
                    tag = timeInPeriod%2
                    if tag == 0:
                        newPoint =  Xp[index/2].reshape((1,d))
                        allDataP[i] = np.concatenate((allDataP[i][1:], newPoint))
                    else:
                        newPoint =  Xq[index/2].reshape((1,d))
                        allDataQ[i] = np.concatenate((allDataQ[i][1:], newPoint))
            
                    globalParams=w0, B0, u0
                    currentData=allDataP[i], allDataQ[i]
                    try:
                        currLeftValue, w2 = getLeftSide(references[i],  
                            globalParams, currentData, R0,alpha)
                        if currLeftValue>R0:
                            violationCounter += 1
                    except:
                        violationCounter += 1
                        currLeftValue=-R0
                    leftValue.append(currLeftValue)
        
                    innerLoopCounter+=1
                    
                #leftValue = len(np.where(leftValue > R0)[0])/float(k)
                
                #leftValue = np.max(leftValue)
                #leftValue /= R0
                
                leftValues.append(violationCounter/k)
                oldNorm = norm(w0)
                S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
                magnitudeDifs.append(np.abs(1-w_norm/oldNorm))
                real = norm(w-w0)
                real /= R0
                reals.append(real)
                
                if real > R0 and violationCounter <= violationThreshold:
                    falseNegative+=1
            
                if violationCounter >violationThreshold and real <= R0:
                    falsePositive+=1
                
                if violationCounter >violationThreshold and real > R0:
                    truePositive+=1
                if real > R0:
                    trueViolationCounter+=1
                    totalTrueViolations+=violationCounter
                else:
                    totalFalseViolations+=violationCounter
                
                if violationCounter >violationThreshold:
                #if real>R0:
                    syncs.append(time+1)
                    for i in range(k):
                        references[i] = getXYS(allDataP[i], allDataQ[i])
                    S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
                    u0 = x0-y0
                    
                    #params.append(time)
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
                    #leftValue = np.mean(leftValue)
                    #leftValue /= R0
                    leftValue = len(np.where(leftValue > R0)[0])/float(k)
                    leftValues.append(violationCounter/k)
                
                    oldNorm = norm(w0)
                    magnitudeDifs.append(np.abs(1-w_norm/oldNorm))
                
                time+=1
        localFalseNegatives.append(falseNegative/(time-trueViolationCounter))
        localFalsePositives.append(falsePositive/(time-trueViolationCounter))
        if trueViolationCounter>0:
            localTruePositives.append(truePositive/trueViolationCounter)
        else:
            localTruePositives.append(1)
    falseNegatives.append(np.mean(localFalseNegatives))
    falsePositives.append(np.mean(localFalsePositives))
    truePositives.append(np.mean(localTruePositives))
plt.plot(params, truePositives)
plt.xlabel('violationThreshold')
plt.ylabel('truePositives')
plt.title(str(dic))

plt.figure()
plt.plot(params, falsePositives)
plt.xlabel('violationThreshold')
plt.ylabel('falsePositisve')
plt.title(str(dic))

plt.figure()
plt.plot(falsePositives, truePositives)
plt.xlabel('falsePositive')
plt.ylabel('truePositives')
plt.title(str(dic))
plt.show() 
    
    
    
    
    
    