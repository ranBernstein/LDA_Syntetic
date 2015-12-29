import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy

d=2
R=2.0
var=1

k=2
L=200
dic={}
timeLength=200*L
NumberOfChangesInDrift = 50
dic['NumberOfChangesInDrift'] = NumberOfChangesInDrift
periods = range(1, 30,3)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
"""
fig = plt.figure()
ax = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(221)
"""
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
for p in periods:
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
#ax1.scatter(periods,cosinesMins)
#ax1.semilogy(periods,cosinesMins)
#dic['dimension'] = d
#plt.plot(periods, syncsNum, label='Syncs' )
#plt.title('Fixed data'+str(dic))
#plt.xlabel('Period size')
#plt.ylabel('Minimum Cosine similarity over time')
#plt.ylim(-0.1, 1.1)

#import cProfile
#cProfile.run('foo()')
        
from numpy import arange,array,ones,linalg
from pylab import plot,show

xi = np.array(periods)
A = array([ xi, ones(len(periods))])
y = cosinesMins
w = linalg.lstsq(A.T,y)[0] # obtaining the parameters
line = w[0]*xi+w[1] # regression line
#plot(xi,line,'r-',xi,y,'o', label=str(w[0])[:8]+'*p+'+str(w[1])[:6])        
msgs = [1.0/p for p in periods]
#errors = 1 - np.array(cosinesMins)
errors = 1 - np.array(line)
ax1.scatter(errors,msgs)
ax1.semilogy(errors,msgs,label='PER')
Ts = cosinesMins
msgs = []
for T in Ts:
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

ax1.scatter(errors,msgs)
ax1.semilogy(errors,msgs, label='DLAD')
ax1.set_title('Fixed Data')
print 'Fixed data, '+str(dic)
#plt.legend().draggable()













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
changeGap = float(timeLength)/NumberOfChangesInDrift
theta = 0
for p in periods:
    cosines=[]
    for time in  range(timeLength):
        if time%changeGap==0:
            print theta, timeLength, changeGap
            theta += np.pi/5
        mu_q[0] = R*np.cos(theta)
        mu_q[1] = R*np.sin(theta)
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
#ax2.scatter(periods,cosinesMins)
#ax2.semilogy(periods,cosinesMins)

from numpy import arange,array,ones,linalg
from pylab import plot,show

xi = np.array(periods)
A = array([ xi, ones(len(periods))])
y = cosinesMins
w = linalg.lstsq(A.T,y)[0] # obtaining the parameters
line = w[0]*xi+w[1] # regression line
#plot(xi,line,'r-',xi,y,'o', label=str(w[0])[:8]+'*p+'+str(w[1])[:6])        
msgs = [1.0/p for p in periods]
#errors = 1 - np.array(cosinesMins)
errors = 1 - np.array(line)
ax2.scatter(errors,msgs)
ax2.semilogy(errors,msgs,label='PER')
Ts = cosinesMins
msgs = []
changeGap = float(timeLength)/NumberOfChangesInDrift
theta = 0
for T in Ts:
    syncs=0.0
    for time in  range(timeLength):
        #changeGap = timeLength/NumberOfChangesInDrift
        #theta = (np.pi/timeLength)*(time - time%changeGap)
        if time%changeGap==0:
            #print theta, timeLength, changeGap
            theta += np.pi/5
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
            syncs+=1
    
    msgs.append(syncs/timeLength)

ax2.scatter(errors,msgs)
ax2.semilogy(errors,msgs, label='DLAD')
ax2.set_title('Drift Data')
print 'Drift data, '+str(dic)

ax2.set_xlabel('Error (1-cosine similarity)')
ax1.set_xlabel('Error (1-cosine similarity)')
ax1.set_ylabel('Normalized messages')
plt.legend().draggable()
plt.show() 
