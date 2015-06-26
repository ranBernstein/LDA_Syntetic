import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample

def norm(v):
    return np.sqrt(np.dot(v.T,v)).item(0)

def cosineSimilarity(v1, v2):
    dot = np.dot(np.array(v1).T[0],np.array(v2).T[0])
    return (dot/norm(v1)/norm(v2)).item(0)

def getCov(X):
    mu = np.matrix(np.mean(X, axis=0))
    X=np.matrix(X)
    numOfObserv = X.shape[0]
    return np.array(1.0/numOfObserv*(X.T*X) - mu.T*mu)

def getR0(w0_norm, T):
    return w0_norm*np.sqrt(1-T**2) 

def FLDInner(cov_p, cov_q, mu_p, mu_q):
    mu_p = np.array(mu_p).T
    mu_q = np.array(mu_q).T
    cov_p = np.array(cov_p)
    cov_q = np.array(cov_q)
    dif = np.matrix(mu_p-mu_q)
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


def getXYS(Xp, Xq):
    x = np.matrix(np.mean(Xp, axis=0)).T
    y = np.matrix(np.mean(Xq, axis=0)).T
    Xp = np.matrix(Xp)
    Xq = np.matrix(Xq)
    n = Xp.shape[0] + Xq.shape[0]
    n_p = Xp.shape[0]
    n_q = Xq.shape[0]
    S = 1.0/n_p*Xp.T*Xp + 1.0/n_q*Xq.T*Xq
    return x,y,S

def checkLocalConstraint(S0, x0, y0, w0_norm, B0_inverted, Xp, Xq, T):
    x,y,S = getXYS(Xp, Xq)
    delta_x = x - x0
    delta_y = y - y0
    Q = delta_x*delta_x.T + delta_y*delta_y.T
    Delta_S = S - S0
    delta = delta_x - delta_y
    L = Delta_S - x0*delta_x.T - delta_x*x0.T - y0*delta_y.T - delta_y*y0.T
    R0 = getR0(w0_norm, T)
    a1 = norm(B0_inverted*delta)
    a2 = w0_norm + R0
    a3 = norm(B0_inverted*L)+norm(B0_inverted*Q)
    a4 = a1 +a2*a3 
    return a4 <= R0

def FLDformula(x,y,S): 
    B = np.matrix(S - x*x.T - y*y.T)
    u = np.matrix(x - y)
    try:
        B_inverted = lin.inv(B)
    except:
        pass
    w = B_inverted*u
    return w, B_inverted

def calcWindowParams(Xp, Xq):
    x,y,S = getXYS(Xp, Xq)
    w, B_inverted = FLDformula(x,y,S)
    w_norm = norm(w)
    return S, x, y, w, w_norm, B_inverted
    
    
    
    
    
    
    
    
    
    