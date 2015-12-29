import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
dic={}
k=200
dic['k']=k

"""
x = np.arange(-100, 100)
p = chernof(x)
plt.plot(x, p)
plt.show()
"""

violationThreshold=190
dic['violationThreshold'] = violationThreshold

alpha=1
#dic['alpha']=alpha
d=2
dic['d']=d
L=100
dic['L']=L
T=0.9
dic['T']=T
periodSize = 10*L
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

allDataP, allDataQ, references  = \
    initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)


#rounds=1000

params = []#range(rounds)
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
time = 0
innerLoopCounter=0
leftValues=[]
magnitudeDifs =[]
reals=[]
syncs=[]
conceptsDrifts=[]
falseNegative=[]
trueViolationCounter=0.
totalTrueViolations=0.
totalFalseViolations=0
for period in range(periodNum):
    theta = 2*np.pi/periodNum*period
    tmp = R*np.cos(theta)
    mu_q[0] = tmp
    mu_q[1] = R*np.sin(theta)
    mu_p = mu_p_0
    Xp = np.random.multivariate_normal(mu_p, cov_p, k*periodSize/2)
    Xq = np.random.multivariate_normal(mu_q, cov_q,  k*periodSize/2)
    conceptsDrifts.append(time)
    for timeInPeriod in range(periodSize):
        params.append(time)
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
            falseNegative.append(time)
    
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
            #leftValue = np.mean(leftValue)
            #leftValue /= R0
            leftValue = len(np.where(leftValue > R0)[0])/float(k)
            leftValues.append(violationCounter/k)
        
            oldNorm = norm(w0)
            magnitudeDifs.append(np.abs(1-w_norm/oldNorm))
        
        time+=1
print falseNegative
dic['falseNegative'] = len(falseNegative)
p_tp = totalTrueViolations/trueViolationCounter/k
p_fp = totalFalseViolations/(time-trueViolationCounter)/k
p_tv = trueViolationCounter/time
dic['p_tp'] = str(p_tp)[:5]
dic['p_fp'] = str(p_fp)[:5]
dic['p_tv'] = str(p_tv)[:5]

dic['syncs'] = len(syncs)
dic['Rounds/syncs_Ratio'] = len(params)/len(syncs)

plt.title(str(dic))
#for sync in syncs:
#    plt.axvline(sync, color='r')

plt.plot(params,leftValues, label='DLDA Bound')
plt.plot(params,reals, label='norm(w-w0)')
plt.plot(params,magnitudeDifs, label='|1 - norm(w)/norm(w0)|')

plt.scatter(syncs, np.ones_like(syncs), c='b', label='Syncs')
conceptsDrifts=conceptsDrifts[1:]
plt.scatter(conceptsDrifts, np.ones_like(conceptsDrifts), c='r', 
            label='Concepts Drifts', marker='x', s=100)
plt.legend().draggable()
plt.xlabel('Round')
plt.ylabel('Error')
plt.title(str(dic))

plt.show() 
    
    
    
    
    
    