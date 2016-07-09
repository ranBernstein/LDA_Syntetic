import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
font = {
    'family': 'normal',
    'weight': 'normal',
    'size': 22
    }
import matplotlib
matplotlib.rc('font', **font) 

def measurDis(w,w0):
    #dic['measure'] = 'cosineSimilarity'
    #return 1-cosineSimilarity(w, w0)
    dic['measure'] = 'norm(w, w0)'
    return norm(w-w0)
d=2
R=2.0
var=1

k=2
L=1000
dic={}
NumberOfChangesInDrift = 50
numberOfWindowsPerDrift = 3
dic['numberOfWindowsPerDrift'] = numberOfWindowsPerDrift
timeLength=NumberOfChangesInDrift*numberOfWindowsPerDrift*L
dic['NumberOfChangesInDrift'] = NumberOfChangesInDrift

periods = range(3, L/10,L/100)
#f, (plt, plt) = plt.subplots(2, 1)#, sharey=True, sharex=True)

mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
mu_q_0[0] = R
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = cov_p_0
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
allDataP, allDataQ, references, _,_ = \
    initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
#cosinesMins = []
R0s=[]
for p in periods:
    real=[]
    for _ in  range(timeLength/p):
        distsParams = mu_p, mu_q, cov_p, cov_q
        for _ in range(p/2):
            for i in range(k):
                updateData(allDataP, allDataQ,i,distsParams,2)
                
            S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
            #c = cosineSimilarity(w0, w) 
            error=measurDis(w,w0)
            real.append(error)
        w0 = w
    R0s.append(np.max(real))
    #cosinesMins.append(min(cosines))
dic['k'] = k
dic['L'] = L
dic['d'] = d
#dic['repeatTime'] = repeatTime
dic['timeLength'] = timeLength
dic['var'] = var


msgs = [1.0/p for p in periods]
errors = R0s
#plt.scatter(errors,msgs)
#plt.semilogy(errors,msgs,label='PER')
plt.figure()
plt.plot(errors,msgs,label='PER', c='g', linestyle='--')
#Ts = cosinesMins
msgs = []
#for T in Ts:
newR0s=[]
for R0 in R0s:
    syncs=0.0
    real =[]
    for _ in  range(timeLength):
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        #R0 = getR0(w0_norm, T)
        globalParams=w0, B0
        violationCounter, globalParams, errors = \
            updateNodes(globalParams, references, data, distsParams, R0)
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
        r = measurDis(w,w0)
        real.append(r)
        if violationCounter > 0:
            syncs+=1
        w0, B0 = globalParams
    newR0s.append(np.max(real))
    msgs.append(syncs/timeLength)

errors=newR0s
#plt.scatter(errors,msgs)
#plt.semilogy(errors,msgs, label='DLAD')
plt.plot(errors,msgs, label='DLAD', c='b')
plt.title('Fixed Data')
print 'Fixed data, '+str(dic)

mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
mu_q_0[0] = R
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = cov_p_0
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
allDataP, allDataQ, references, _,_ = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
changeGap = float(timeLength)/NumberOfChangesInDrift
R0s = []
for p in periods:
    theta = 0
    real=[]
    for time in  range(timeLength):
        if time%changeGap==0:
            theta += np.pi/2
        mu_q[0] = R*np.cos(theta)
        mu_q[1] = R*np.sin(theta)
        distsParams = mu_p, mu_q, cov_p, cov_q
        for i in range(k):
            updateData(allDataP, allDataQ,i,distsParams,p) 
        S, x, y, w, w_norm, B_inverted = calcWindowParams2D(allDataP, allDataQ)
        #c = cosineSimilarity(w0, w) 
        c=measurDis(w, w0)
        w0 = w
        real.append(c)
    R0s.append(np.max(real))
dic['k'] = k
dic['L'] = L
dic['d'] = d
#dic['repeatTime'] = repeatTime
dic['timeLength'] = timeLength
dic['var'] = var

#errors = 1 - np.array(cosinesMins)
#errors = 1 - np.array(line)
errors = R0s
msgs = [1.0/p for p in periods]
#plt.scatter(errors,msgs)
#plt.semilogy(errors,msgs,label='PER')
plt.figure()
plt.plot(errors,msgs,label='PER', c='g', linestyle='--')
#Ts = cosinesMins
msgs = []
#for T in Ts:
newR0s=[]
for R0 in R0s:
    theta = 0
    syncs=0.0
    real = []
    for time in  range(timeLength):
        #changeGap = timeLength/NumberOfChangesInDrift
        #theta = (np.pi/timeLength)*(time - time%changeGap)
        if time%changeGap==0:
            theta += np.pi/2
        mu_q[0] = R*np.cos(theta)
        mu_q[1] = R*np.sin(theta)
        data = allDataP, allDataQ
        distsParams = mu_p, mu_q, cov_p, cov_q
        globalParams =  w0, B0, u0
        violationCounter, globalParams, errors \
            = updateNodes(globalParams, references,  
                          data , distsParams, R0)
        S, x, y, w, w_norm, B = calcWindowParams2D(allDataP, allDataQ)
        r=measurDis(w, w0)
        real.append(r)
        if violationCounter > 0:
            syncs+=1
        w0, B0, u0 = globalParams
    newR0s.append(np.max(real))
    msgs.append(syncs/timeLength)
errors=newR0s
#plt.scatter(errors,msgs)
#plt.semilogy(errors,msgs, label='DLAD')
plt.plot(errors,msgs, label='DLDA', c='b')
#plt.title('Drift Data')
#plt.title('Drift Data')
plt.xlabel('Model Drift')
#plt.xlabel('Error')
plt.ylabel('Normalized messages')
plt.legend().draggable()
#plt.title(str(dic))
plt.show() 
