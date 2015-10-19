import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy

d=2
R=2.0
var=1
T=0.997
k=10
L=1000
dic={}
timeLength=10*L
p=100

mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
mu_q_0[0] = R
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = cov_p_0
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
cosines = []
cosinesPer = []
time=0
allDataP, allDataQ, references = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
S0_per, x0_per, y0_per, w0_per, w0_norm_per, \
    B0_inverted_per = calcWindowParams2D(allDataP, allDataQ)
periodDataP = np.zeros((k,p,d))
periodDataQ = np.zeros((k,p,d))
params =  range(timeLength)
syncs = []
for time in params:
    changeGap = timeLength/4
    theta = (np.pi/timeLength)*(time - time%changeGap)
    mu_q[0] = R*np.cos(theta)
    mu_q[1] = R*np.sin(theta)
    currP = time%p
    violationCounter = 0
    for i in range(k):        
        newP = np.random.multivariate_normal(mu_p, cov_p,1)
        allDataP[i] = np.concatenate((allDataP[i][1:], newP))
        
        newQ = np.random.multivariate_normal(mu_q, cov_q,1)
        allDataQ[i] = np.concatenate((allDataQ[i][1:], newQ))

        x0_i, y0_i, S0_i = references[i]
        isInSafeZone = checkLocalConstraint(S0_i, x0_i, y0_i, w0_norm, \
            B0_inverted, allDataP[i], allDataQ[i], T)
        if not isInSafeZone:
            violationCounter += 1
        
        if currP == p-1:
            S0_per, x0_per, y0_per, w0_per, w0_norm_per, \
                B0_inverted_per = calcWindowParams2D(allDataP, allDataQ)
    if violationCounter > 0:
        syncs.append(time)
        for i in range(k):
            referenceParams = getXYS(allDataP[i], allDataQ[i])
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ) 

        #updateData(allDataP, allDataQ,i,distsParams,p) 
    S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
    c = cosineSimilarity(w0, w) 
    c_per = cosineSimilarity(w0_per, w)
    cosines.append(c)
    cosinesPer.append(c_per)    
dic['Nodes num'] = k
dic['Window size'] = L
dic['Dimension'] = d
dic['PER period'] = p
dic['Threshold'] = T
dic['Our syncs'] = len(syncs)
dic['PER syncs'] = timeLength/p
font = {
    'family': 'normal',
    'weight': 'normal',
    'size': 22
    }
import matplotlib
matplotlib.rc('font', **font) 
#dic['repeatTime'] = repeatTime
#dic['timeLength'] = timeLength
#dic['Var'] = var 
cosines = 1-np.array(cosines)
cosinesPer = 1-np.array(cosinesPer)
plt.plot(params,cosines, label='DLDA')
plt.plot(params,cosinesPer, label='PER')
plt.axhline(y=1-T, label='Monitoring Error Threshold', linestyle='--', color='r')
#plt.title(str(dic))
import inspect, os
file = (inspect.getfile(inspect.currentframe())).split('\\')[-1]
print file, str(dic)
plt.xlabel('Round')
plt.ylabel('Model error')   
plt.legend().draggable()
plt.show() 