import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *

mu_p_0 = [2,2]
mu_q_0 = [-2,-2]
cov_p_0 = [[1,1],\
         [1,2]]
cov_q_0 = [[2,-1],\
         [-1,1]]
k=1
T=0.3
n_p = 50
n_q = 50
n = n_p + n_q

params = np.linspace(0, 1, 20)
res=[]
coses=[]
isInAdmissibleRegions=[]

#Data before last sync
Xp_0 =  np.random.multivariate_normal(mu_p_0, cov_p_0, n_p*k)
Xp_0 = np.matrix(Xp_0)
Xq_0 =  np.random.multivariate_normal(mu_q_0, cov_q_0, n_q*k)
Xq_0 = np.matrix(Xq_0)

S0 = 1.0/n_p/k*Xp_0.T*Xp_0 + 1.0/n_q/k*Xq_0.T*Xq_0
x0 = np.mean(Xp_0, axis=0).T
y0 = np.mean(Xq_0, axis=0).T
B0 = S0 - x0*x0.T - y0*y0.T
u0 = x0 - y0
B0_inverted = lin.inv(B0)
w0 = B0_inverted*u0
w0_norm = norm(w0)

for p in params:       
    #Data after sync
    cov_p=cov_p_0
    cov_p[0][0] += p
    mu_p=mu_p_0
    mu_p[0] -=p
    Xp =  np.random.multivariate_normal(mu_p, cov_p, n_p*k)
    Xp = np.matrix(Xp)
    
    mu_q = mu_q_0
    mu_q[1] += p
    cov_q = cov_q_0
    cov_q[1][1] += p
    Xq =  np.random.multivariate_normal(mu_q_0, cov_q_0, n_q*k)
    Xq = np.matrix(Xq)
    
    w, c = FLD(Xp, Xq)
    cos = cosineSimilarity(w, w0)
    coses.append(cos)
    isInAdmissibleRegion = (cos > T)
    if isInAdmissibleRegion:
        isInAdmissibleRegions.append(1)
    else:
        isInAdmissibleRegions.append(0)
    #print "Is new data in admissible region? ", isInAdmissibleRegion
    
    counter=0.0
    for i in range(k):
        Xp_i =  Xp[i*n_p:(i+1)*n_p]
        Xq_i =  Xq[i*n_q:(i+1)*n_q]   
        isInSafeZone = checkLocalConstraint(S0, x0, y0, \
                        w0_norm, B0_inverted, Xp_i, Xq_i, T)
        if isInSafeZone and not isInAdmissibleRegion:
            raise "We have a bug"
        if isInSafeZone:
            counter=counter+1
    res.append(counter/k)
    
plt.plot(params, res, label='Our condition')
plt.plot(params, isInAdmissibleRegions, label='cosine similarity')
#plt.semilogx(ns, coses, label='Cosine Similarity of the whole data')
dic={}
dic['k'] = k
dic['T'] = T 
plt.title('Comparison between our condition (one node) to cosine similarity, '+str(dic))
plt.xlabel('Drift magnitude')
plt.ylabel('Is the condition satisfied')
plt.ylim(-0.25,1.25)
plt.legend().draggable()

plt.show()