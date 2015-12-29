import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *

mu_p_0 = [-2,-2]
mu_q_0 = [2,2]
cov_p_0 = [[1,1],\
         [1,2]]
cov_q_0 = [[2,-1],\
         [-1,1]]

w0, c0 = FLDInner(cov_p_0, cov_q_0, mu_p_0, mu_q_0)
print 'w0', w0
print 'c0', c0
n=100
k=10



totalDiffs0=[]
totalCosines0=[]
totalNormalizeddiffs0=[]

totalDiffsTrue=[]
totalCosinesTrue=[]
totalNormalizeddiffsTrue=[] 

maxDiffsTrue=[]
maxCosinesTrue=[]
maxNormalizeddiffsTrue=[] 

w0_wT_Normalizeddiffs=[]
drifts = np.linspace(0, 1, 10)
for drift in drifts: 
    diffs0=[]
    cosines0=[]
    normalizeddiffs0=[]
    
    diffsTrue=[]
    cosinesTrue=[]
    normalizeddiffsTrue=[] 
    
    mu_p = mu_p_0
    mu_q = mu_q_0
    cov_p = cov_p_0
    cov_q = cov_q_0
    cov_p[0][0] = cov_p[0][0]+drift
    Xp =  np.random.multivariate_normal(mu_p, cov_p, n*k)
    Xq =  np.random.multivariate_normal(mu_q, cov_q, n*k)
    w_true, c_true = FLD(Xp, Xq)
    
    w0_wT_Normalizeddiffs.append(norm(w_true/norm(w_true) - w0/norm(w0)))
    maxDif, maxCos, maxNormilizedDif = -np.inf, -np.inf, -np.inf
    for i in range(k):
        Xp_i =  Xp[i*n:(i+1)*n]
        Xq_i =  Xq[i*n:(i+1)*n]
        w, c = FLD(Xp_i, Xq_i)
        
        diffs0.append(norm(w - w0))
        cosines0.append(cosineSimilarity(w,w0))
        normalizeddiffs0.append(norm(w/norm(w) - w0/norm(w0)))
        
        diff = norm(w - w_true)
        diffsTrue.append(diff)
        cos = cosineSimilarity(w,w_true)
        cosinesTrue.append(cos)
        nor = norm(w/norm(w) - w_true/norm(w_true))
        normalizeddiffsTrue.append(nor)
        
        maxDif, maxCos, maxNormilizedDif = max(maxDif, diff),  max(maxCos,cos),  max(maxNormilizedDif, nor)
    
    totalDiffs0.append(np.mean(diffs0))
    totalCosines0.append(np.mean(cosines0))
    totalNormalizeddiffs0.append(np.mean(normalizeddiffs0))
    
    totalDiffsTrue.append(np.mean(diffsTrue))
    totalCosinesTrue.append(np.mean(cosinesTrue))
    totalNormalizeddiffsTrue.append(np.mean(normalizeddiffsTrue))  
    
    maxDiffsTrue.append(maxDif)
    maxCosinesTrue.append(maxCos)
    maxNormalizeddiffsTrue.append(maxNormilizedDif)

plt.figure()
plt.plot(drifts, totalDiffs0, label="norm(w - w0) Average")
plt.plot(drifts, totalCosines0, label="cosineSimilarity(w,w0) Average")
plt.plot(drifts, totalNormalizeddiffs0, label="norm(w/norm(w) - w0/norm(w0)) Average")
plt.legend().draggable()

plt.figure()
plt.plot(drifts, totalDiffsTrue, label="norm(w - w_true) Average")
plt.plot(drifts, totalCosinesTrue, label="cosineSimilarity(w,w_true) Average")
plt.plot(drifts, totalNormalizeddiffsTrue, label="norm(w/norm(w) - w_true/norm(w_true)) Average")
plt.legend().draggable()

plt.figure()
plt.plot(drifts, maxDiffsTrue, label="norm(w - w_true) Max")
plt.plot(drifts, maxCosinesTrue, label="cosineSimilarity(w,w_true) Max")
plt.plot(drifts, maxNormalizeddiffsTrue, label="norm(w/norm(w) - w_true/norm(w_true)) Max")
plt.legend().draggable()

plt.figure()
plt.xlabel("norm(w_true/norm(w_true) - w0/norm(w0))")
plt.plot(w0_wT_Normalizeddiffs, totalDiffs0, label="norm(w - w0) Average")
plt.plot(w0_wT_Normalizeddiffs, totalCosines0, label="cosineSimilarity(w,w0) Average")
plt.plot(w0_wT_Normalizeddiffs, totalNormalizeddiffs0, label="norm(w/norm(w) - w0/norm(w0)) Average")
plt.legend().draggable()

"""
plt.figure()
plt.title("norm(w - w0)")
plt.plot(drifts, totalDiffs0)

plt.figure()
plt.title("cosineSimilarity(w,w0)")
plt.plot(drifts, totalCosines0)

plt.figure()
plt.title("norm(w/norm(w) - w0/norm(w0))")
plt.plot(drifts, totalNormalizeddiffs0)

plt.figure()
plt.title("norm(w - w_true) Average")
plt.plot(drifts, totalDiffsTrue)

plt.figure()
plt.title("cosineSimilarity(w,w_true) Average")
plt.plot(drifts, totalCosines0)

plt.figure()
plt.title("norm(w/norm(w) - w_true/norm(w_true)) Average")
plt.plot(drifts, totalNormalizeddiffs0)

plt.figure()
plt.title("norm(w - w_true) Max")
plt.plot(drifts, maxDiffsTrue)

plt.figure()
plt.title("cosineSimilarity(w,w_true) Max")
plt.plot(drifts, maxCosinesTrue)

plt.figure()
plt.title("norm(w/norm(w) - w_true/norm(w_true)) Max")
plt.plot(drifts, maxNormalizeddiffsTrue)
"""
plt.show() 
    
    
    
    
    
    
    
    
