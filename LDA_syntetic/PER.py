import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy

d=2
R=2.0
var=1

k=1
L=200
dic={}
#beforSyncLength = L
timeLength=1000#2*L
#T=0.4
#params = np.linspace(0,1,10)
params = range(1,150, 15)

mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
mu_q_0[0] = R
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = cov_p_0
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
allDataP, allDataQ, references = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ)
cosinesMins = []
for p in params:
    cosines=[]
    for _ in  range(timeLength):
        distsParams = mu_p, mu_q, cov_p, cov_q
        for i in range(k):
            updateData(allDataP, allDataQ,i,distsParams,p) 
        S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        c = cosineSimilarity(w0, w) 
        w0 = w
        cosines.append(c)
    cosinesMins.append(min(cosines))
dic['k'] = k
dic['L'] = L
dic['d'] = d
#dic['repeatTime'] = repeatTime
dic['timeLength'] = timeLength
dic['var'] = var
plt.scatter(params,cosinesMins)
#plt.plot(params,cosinesMins)
#dic['dimension'] = d
#plt.plot(params, syncsNum, label='Syncs' )
plt.title('Fixed data'+str(dic))
plt.xlabel('Period size')
plt.ylabel('Minimum Cosine similarity over time')
#plt.ylim(-0.1, 1.1)

#import cProfile
#cProfile.run('foo()')
        
from numpy import arange,array,ones,linalg
from pylab import plot,show

xi = np.array(params)
A = array([ xi, ones(len(params))])
# linearly generated sequence
y = cosinesMins
w = linalg.lstsq(A.T,y)[0] # obtaining the parameters

# plotting the line
line = w[0]*xi+w[1] # regression line
plot(xi,line,'r-',xi,y,'o', label=str(w[0])[:8]+'*p+'+str(w[1])[:6])        
plt.legend().draggable()

plt.figure()
msgs = [1.0/p for p in params]
#errors = 1 - np.array(cosinesMins)
errors = 1 - np.array(line)
plt.scatter(errors,msgs)
plt.semilogy(errors,msgs,label='PER')

#params = np.linspace(min(errors), max(errors), 10)
params = cosinesMins
msgs = []
for T in params:
    syncs=0.0
    for _ in  range(timeLength):
        globalParams = k, T, w0, w0_norm, B0_inverted
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        violationCounter, globalParams = updateNodes(globalParams, references, 
                                                     data , distsParams, True)
        k, T, w0, w0_norm, B0_inverted = globalParams
        #S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        if violationCounter > 0:
            syncs+=1
    
    msgs.append(syncs/timeLength)

plt.scatter(errors,msgs)
plt.semilogy(errors,msgs, label='Ours')
plt.legend().draggable()
plt.xlabel('Error (1-cosine similarity)')
plt.ylabel('Normalized messages')
plt.title('Fixed data, '+str(dic))
plt.show() 