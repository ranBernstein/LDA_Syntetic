import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample

def norm(v):
    return np.sqrt(np.dot(v,v))

def cosineSimilarity(v1, v2):
    return np.dot(v1,v2)/norm(v1)/norm(v2)

def getCov(X):
    mu = np.matrix(np.mean(X, axis=0))
    X=np.matrix(X)
    numOfObserv = X.shape[0]
    return np.array(1.0/numOfObserv*(X.T*X) - mu.T*mu)


def FLDInner(cov_p, cov_q, mu_p, mu_q):
    mu_p = np.array(mu_p)
    mu_q = np.array(mu_q)
    cov_p = np.array(cov_p)
    cov_q = np.array(cov_q)
    dif = np.matrix(mu_p-mu_q).T
    S = cov_p+cov_q
    w = lin.inv(S)*dif
    dif=np.array(dif.T)[0]
    tempW =np.array(w.T)[0] 
    c = 0.5*np.dot(tempW, mu_p+mu_q)
    w=np.array(w).T[0]
    return w, c

def FLD(Xp, Xq):
    mu_p = np.mean(Xp, axis=0)
    mu_q = np.mean(Xq, axis=0)
    cov_p = getCov(Xp)
    cov_q = getCov(Xq)
    return FLDInner(cov_p, cov_q, mu_p, mu_q)

def simulateNode(mu_p, cov_p, mu_q, cov_q, n):
    Xp =  np.random.multivariate_normal(mu_p, cov_p, n)
    Xq =  np.random.multivariate_normal(mu_q, cov_q, n)
    return FLD(Xp, Xq)
    
mu_p = [-2,-2]
mu_q = [2,2]
cov_p = [[1,1],\
         [1,2]]
cov_q = [[2,-1],\
         [-1,1]]
w0, c0 = FLDInner(cov_p, cov_q, mu_p, mu_q)
print 'w0', w0
print 'c0', c0
n=200
k=1000

Xp =  np.random.multivariate_normal(mu_p, cov_p, n)
x,y = Xp.T
plt.plot(x,y,'x'); plt.axis('equal'); 
Xq =  np.random.multivariate_normal(mu_q, cov_q, n)
x,y = Xq.T
plt.plot(x,y,'x'); plt.axis('equal'); 

w, c = FLD(Xp, Xq)
x = np.concatenate((Xp, Xq))
xx = np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01)
yy = - (w[0] * xx + c) / w[1]
print 'w:', w
print 'c', c
plot2 = plt.plot(xx, yy, '--k')


ns=range(1, n,10)
totalDiffs=[]
totalCosines=[]
totalNorms=[]
totalNormalizeddiffs=[]
for i in ns:
    diffs=[]
    cosines=[]
    norms=[]
    normalizeddiffs=[]
    for j in range(k):
        w, c = simulateNode(mu_p, cov_p, mu_q, cov_q, n)
        diffs.append(norm(w - w0))
        cosines.append(cosineSimilarity(w,w0))
        norms.append(norm(w))
        normalizeddiffs.append(np.mean(norm(w/norm(w) - w0/norm(w0))))
    totalDiffs.append(np.mean(diffs))
    totalCosines.append(np.mean(cosines))
    totalNorms.append(np.mean(norms))
    totalNormalizeddiffs.append(np.mean(normalizeddiffs))
plt.figure()
plt.title("norm(w - w0)")
plt.plot(ns, totalDiffs)

plt.figure()
plt.title("cosineSimilarity(w,w0)")
plt.plot(ns, totalCosines)

plt.figure()
plt.title("norm(w)")
plt.plot(ns, totalNorms)

plt.figure()
plt.title("norm(w/norm(w) - w0/norm(w0))")
plt.plot(ns, totalNormalizeddiffs)
plt.show()

    
