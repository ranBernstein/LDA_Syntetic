import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
import matplotlib.cm as cm
#d=20
R=2.0
var=1
alpha=1
beta=1
k=1
violationThreshold=0
L=100
dic={}
#beforSyncLength = L
timeLength=1000#2*L
T=0.4

for m,d in enumerate([2,15,30, 60]):#enumerate(range(2,350,10)):
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
    S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
    u0 = x0-y0
    violationCounters=[]
    cosines = []
    violationCounter = 0.
    singularityCounter = 0.
    CScounter = 0.
    errors={}
    for t in params:
        globalParams=w0, B0, u0
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        R0 = getR0(w0_norm, T)
        leftValue = []
        for i in range(k):
            #referenceParams = references[i]
            dataP = allDataP[i]
            dataQ = allDataQ[i]
            dataP, dataQ = (dataP[1:], dataQ[1:]) 
            newP = np.random.multivariate_normal(mu_p, cov_p,1)
            newQ = np.random.multivariate_normal(mu_q, cov_q,1)
            dataP = np.concatenate((dataP, newP))
            dataQ = np.concatenate((dataQ, newQ))
            allDataP[i] = dataP
            allDataQ[i] = dataQ
            currentData=allDataP[i], allDataQ[i]
            try:
                currLeftValue, w2 = getLeftSide(references[i],  globalParams, 
                                                currentData, R0*beta,alpha)
                if currLeftValue>R0:
                    #print 'Regular Violation'
                    violationCounter += 1
            except Exception as e:
                msg =  e.message
                errors[msg]=errors.get(msg,0)+1
                #print 'Singularity'
                violationCounter += 1
                currLeftValue=-R0
            leftValue.append(currLeftValue)
        w0, B0, u0 = globalParams
        S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        violationCounters.append(violationCounter)
        if violationCounter >violationThreshold:
            for i in range(k):
                referenceParams = getXYS(allDataP[i], allDataQ[i])
                references[i] = referenceParams
            S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
            u0 = x0-y0 
    #plt.figure()
    #plt.scatter(errors.keys(), errors.values())
    #params2 = np.nonzero(violationCounters)[0]
    #plt.scatter(params2, np.array(violationCounters)[params2], 
                #label='sync, d='+str(d), marker=markers[m])
    #plt.plot(params, cosines, label='Cosine similarity, d='+str(d))
    print 'Dimension', d
    print 'Experiment Length', timeLength
    print 'Total violations', violationCounter
    print 'Becketing:'
    totalOther=0
    for i,(key,v) in enumerate(errors.items()):
        print key+':', v
        #plt.scatter([d],[v], c=cm.hot(float(i)/len(errors)), label=key)
        totalOther += v
    #plt.scatter([d],[violationCounter-totalOther], c='blue', label="Delta -> L+M")
    print 'Breaking Covariance drift into linear and quadratic parts:', \
            violationCounter - totalOther
    print ""
    print ""

#plt.figure()
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
#plt.show()   