import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
from Orange.clustering.hierarchical import matplotlib

d=2
R=2.0
var=1

k=2
#L=1000
dic={}
#timeLength=10*L

T=0.9
mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
mu_q_0[0] = R
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = cov_p_0
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
params = range(20,500, 25)
syncs = []
for L in params:
    timeLength=10*L
    allDataP, allDataQ, references,_,_ = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
    S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
    syncsCounter=0.0
    for _ in  range(timeLength):
        globalParams = w0, B0
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        violationCounter, globalParams, errors = \
            updateNodes(globalParams, references, data , distsParams, True)
        w0, B0 = globalParams
        #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        if violationCounter > 0:
            syncsCounter+=1
    syncs.append(syncsCounter/timeLength)

driftSyncs = []
for L in params:
    allDataP, allDataQ, references,_,_ = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
    S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
    syncsCounter=0.0
    for time in  range(timeLength):
        changeGap = timeLength/4
        theta = (np.pi/timeLength)*(time - time%changeGap)
        mu_q[0] = R*np.cos(theta)
        mu_q[1] = R*np.sin(theta)
        
        globalParams = w0, B0 
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        violationCounter, globalParams, errors = \
            updateNodes(globalParams, references, data , distsParams, True)
        w0, B0  = globalParams
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
#plt.scatter(params,syncs)
#plt.scatter(params,driftSyncs)
#plt.semilogy(params,syncs, label='Ours')
#plt.plot(params,syncs, label='Fixed')
plt.plot(params,driftSyncs, label='Drift')#, c='g', linestyle='--')
#plt.legend().draggable()
plt.xlabel('Window Size')
plt.ylabel('Norm. Msgs')
import inspect, os
file = (inspect.getfile(inspect.currentframe())).split('\\')[-1]
print file+str(dic)
#plt.title(file+str(dic))
plt.show()


 