import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
from Orange.clustering.hierarchical import matplotlib

d=2
R=2.0
#var=1
k=2
L=100
dic={}
T=0.99

timeLength=100*L
params = np.logspace(-0.3,-1,6)
syncs = []
for sd in params:
    mu_p_0 = np.zeros((d,))
    mu_q_0 = np.zeros((d,))
    mu_q_0[0] = R
    cov_p_0 = (sd**2)*np.diag(np.ones((d,)))
    cov_q_0 = cov_p_0
    mu_p = copy.copy(mu_p_0)
    mu_q = copy.copy(mu_q_0)
    cov_p = copy.copy(cov_p_0)
    cov_q = copy.copy(cov_q_0)
    allDataP, allDataQ, references = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
    S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
    syncsCounter=0.0
    for _ in  range(timeLength):
        globalParams = k, T, w0, w0_norm, B0_inverted
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        violationCounter, globalParams = updateNodes(globalParams, references, 
                                                     data , distsParams, True)
        k, T, w0, w0_norm, B0_inverted = globalParams
        #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        if violationCounter > 0:
            syncsCounter+=1
    syncs.append(syncsCounter/timeLength)

driftSyncs = []
for sd in params:
    mu_p_0 = np.zeros((d,))
    mu_q_0 = np.zeros((d,))
    mu_q_0[0] = R
    cov_p_0 = (sd**2)*np.diag(np.ones((d,)))
    cov_q_0 = cov_p_0
    mu_p = copy.copy(mu_p_0)
    mu_q = copy.copy(mu_q_0)
    cov_p = copy.copy(cov_p_0)
    cov_q = copy.copy(cov_q_0)
    allDataP, allDataQ, references = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
    S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
    syncsCounter=0.0
    for time in  range(timeLength):
        changeGap = timeLength/4
        theta = (np.pi/timeLength)*(time - time%changeGap)
        mu_q[0] = R*np.cos(theta)
        mu_q[1] = R*np.sin(theta)
        
        globalParams = k, T, w0, w0_norm, B0_inverted
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        violationCounter, globalParams = updateNodes(globalParams, references, 
                                                     data , distsParams, True)
        k, T, w0, w0_norm, B0_inverted = globalParams
        #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        if violationCounter > 0:
            syncsCounter+=1
    driftSyncs.append(syncsCounter/timeLength)
    
dic['Window Size'] = L
dic['Dimension'] = d
dic['Threshold'] = T
dic['Time Length'] = '10*L'#timeLength
font = {
    'family': 'normal',
    'weight': 'normal',
    'size': 22
    }
matplotlib.rc('font', **font)
plt.scatter(params,syncs)
plt.scatter(params,driftSyncs)
#plt.semilogy(params,syncs, label='Ours')
plt.semilogx(params,syncs, label='Fixed')
plt.semilogx(params,driftSyncs, label='Drift')
plt.legend().draggable()
plt.xlabel('Noise standard deviation')
plt.ylabel('Norm. Msgs')
import inspect, os
file = (inspect.getfile(inspect.currentframe())).split('\\')[-1]
print file+str(dic)
#plt.title(file+str(dic))
plt.show()


 