import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy

#d=20
R=2.0
var=1

k=1
L=400
dic={}
#beforSyncLength = L
timeLength=1000#2*L
T=0.4


#repeatTime = 20
#def foo():
#violationCounters= np.zeros((len(params),))
#cosines = np.zeros((len(params),))
#violationsNum = np.zeros((len(params),))
markers=['<','>']
for m,d in enumerate(range(2,350,10)):#enumerate([2,4,8,10,15,20]):
    mu_p_0 = np.zeros((d,))
    mu_q_0 = np.zeros((d,))
    mu_q_0[0] = R
    cov_p_0 = var*np.diag(np.ones((d,)))
    cov_q_0 = cov_p_0
    params = range(timeLength)
    mu_p = copy.copy(mu_p_0)
    mu_q = copy.copy(mu_q_0)
    cov_p = copy.copy(cov_p_0)
    cov_q = copy.copy(cov_q_0)
    allDataP, allDataQ, references = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
    S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
    violationCounters=[]
    cosines = []
    for t in params:
        globalParams = k, T, w0, w0_norm, B0_inverted
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        violationCounter, globalParams = updateNodes(globalParams, references, 
                                                     data , distsParams, True)
        k, T, w0, w0_norm, B0_inverted = globalParams
        S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        c = cosineSimilarity(w0, w) 
        cosines.append(c)
        violationCounters.append(violationCounter)
        
    params2 = np.nonzero(violationCounters)[0]
    #plt.scatter(params2, np.array(violationCounters)[params2], 
                #label='sync, d='+str(d), marker=markers[m])
    #plt.plot(params, cosines, label='Cosine similarity, d='+str(d))
    
    plt.scatter([d],[float(len(params2))/timeLength])

dic['k'] = k
dic['L'] = L
#dic['repeatTime'] = repeatTime
dic['timeLength'] = timeLength
dic['T(In radians)'] = str(T)+'('+str(np.arccos(T))[:4]+')'
dic['var'] = var
#dic['dimension'] = d
#plt.plot(params, syncsNum, label='Syncs' )
plt.title('Fixed data'+str(dic))
plt.xlabel('Dimension')
plt.ylabel('Probability for sync')
#plt.legend().draggable()
#plt.ylim(-0.1, 1.1)

#import cProfile
#cProfile.run('foo()')
plt.show()   