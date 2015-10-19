import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from numpy import linalg as LA

def norm(v):
    if len(v.shape) >1 and v.shape[1] >1:
        raise "Not a vector"
    return np.sqrt(np.dot(v.T,v)).item(0)
def normOperator(A):
    A= np.matrix(A)
    AAT = A*A.T
    w, v = LA.eig(AAT)
    largestEigen = max(w)
    return np.sqrt(largestEigen)

def cosineSimilarity(v1, v2):
    if len(v1)!=len(v2):
        raise "Bad argumant"
    dot = np.dot(np.array(v1).T[0],np.array(v2).T[0])
    return (dot/(norm(v1)*norm(v2))).item(0)

def getCov(X):
    mu = np.matrix(np.mean(X, axis=0))
    X=np.matrix(X)
    numOfObserv = X.shape[0]
    return np.array(1.0/numOfObserv*(X.T*X) - mu.T*mu)

def getR0(w0_norm, T):
    #return w0_norm*np.sin(np.arccos(T))
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

def checkLocalConstraint(localParams,  globalParams, currentData, R0):
    a4, waste = getLeftSide(localParams,  globalParams, currentData, R0)
    return a4 <= R0

def getLeftSide(localParams,  globalParams, currentData, R0, alpha=1):
    x0, y0, S0 = localParams
    w0, B0, u0, w = globalParams
    Xp, Xq = currentData
    w0_norm = norm(w0)
    #R0 = getR0(w0_norm, T)
    x_i,y_i,S_i = getXYS(Xp, Xq)
    #B0  = np.matrix(S0  - x0*x0.T - y0*y0.T)
    B0_inverted = lin.inv(B0)
    delta_x = x_i - x0
    delta_y = y_i - y0
    Q = delta_x*delta_x.T + delta_y*delta_y.T
    Delta_S = S_i - S0
    delta = delta_x - delta_y
    L = Delta_S - x0*delta_x.T - delta_x*x0.T - y0*delta_y.T - delta_y*y0.T
    a1 = norm(B0_inverted*delta)
    a2 = alpha*w0_norm + R0
    a3 = normOperator(B0_inverted*L)+normOperator(B0_inverted*Q)
    
    Delta = Q + L
    B = np.matrix(S_i - x_i*x_i.T - y_i*y_i.T)
    #u0 = x0-y0
    E1 = norm(lin.inv(B0+Delta)*delta)
    E2 = norm((lin.inv(B0+Delta)-B0_inverted)*u0)
    wastes = np.zeros(6)
    wastes[0] = E1 + E2 - norm(w- w0) 
    a6 =     normOperator(B0_inverted*Delta)
    denominator = 1- a6
    a5 = norm(B0_inverted*Delta*w0)
    wastes[4] = a3 - a6
    if denominator > 0:
        #wastes[3] = (a6*w0_norm - a5)/(R0*denominator)
        wastes[3] = a6*w0_norm/a5
        wastes[1] = \
            norm(B0_inverted*delta)/denominator - E1
        waste3 = a5/denominator - E2
        a4 = a1 +a2*a3 
    else:
        wastes[1] = R0
        wastes[2] = R0
        wastes[3] = R0
        return R0+1, wastes
    if wastes[0] < 0:
        pass
    #wastes = [waste1/R0, waste2/R0, waste3/R0, waste4, waste5*(R0+w0_norm)/R0]
    wastes[5] = np.sum(wastes)
    return a4, wastes

def FLDformula(x,y,S): 
    B = np.matrix(S - x*x.T - y*y.T)
    u = np.matrix(x - y)
    B_inverted = lin.inv(B)
    w = B_inverted*u
    return w, B

def calcWindowParams(Xp, Xq):
    x,y,S = getXYS(Xp, Xq)
    w, B = FLDformula(x,y,S)
    w_norm = norm(w)
    return S, x, y, w, w_norm, B

def calcWindowParams2D(allDataP, allDataQ):
    k=allDataP.shape[0]
    Lp=allDataP.shape[1]
    Lq=allDataQ.shape[1]
    d=allDataP.shape[2]
    allDataPStacked = allDataP.reshape((k*Lp,d))
    allDataQStacked = allDataQ.reshape((k*Lq,d))
    return calcWindowParams(allDataPStacked, allDataQStacked)
    
def initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0):
    allDataP = np.zeros((k,L,d))
    allDataQ = np.zeros((k,L,d))
    references=[]
    for i in range(k):
        Xp_0 = np.random.multivariate_normal(mu_p_0, cov_p_0, L)
        Xq_0 = np.random.multivariate_normal(mu_q_0, cov_q_0, L)
        allDataP[i] = Xp_0
        allDataQ[i] = Xq_0
        referenceParams = getXYS(Xp_0, Xq_0)
        references.append(referenceParams)
    return allDataP, allDataQ, references

def updateNodes(globalParams, references, data , distsParams, sync):
    k, T, w0, w0_norm, B0_inverted = globalParams
    allDataP, allDataQ = data
    mu_p, mu_q, cov_p, cov_q = distsParams
    violationCounter = 0
    for i in range(k):
        referenceParams = references[i]
        dataP = allDataP[i]
        dataQ = allDataQ[i]
        dataP, dataQ = (dataP[1:], dataQ[1:]) 
        newP = np.random.multivariate_normal(mu_p, cov_p,1)
        newQ = np.random.multivariate_normal(mu_q, cov_q,1)
        dataP = np.concatenate((dataP, newP))
        dataQ = np.concatenate((dataQ, newQ))
        allDataP[i] = dataP
        allDataQ[i] = dataQ

        x0_i, y0_i, S0_i = referenceParams
        isInSafeZone = checkLocalConstraint(S0_i, x0_i, y0_i, w0_norm, \
            B0_inverted, dataP, dataQ, T)
        if not isInSafeZone:
            violationCounter += 1
    if sync and violationCounter > 0:
        for i in range(k):
            dataP = allDataP[i]
            dataQ = allDataQ[i]
            referenceParams = getXYS(dataP, dataQ)
            references[i] = referenceParams
        S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams2D(allDataP, allDataQ) 
    globalParams = k, T, w0, w0_norm, B0_inverted
    return violationCounter, globalParams

def updateData(allDataP, allDataQ,i,distsParams,p):
    mu_p, mu_q, cov_p, cov_q = distsParams
    dataP = allDataP[i]
    dataQ = allDataQ[i]
    dataP, dataQ = (dataP[p:], dataQ[p:])
    try: 
        newP = np.random.multivariate_normal(mu_p, cov_p,p)
    except:
        pass
    newQ = np.random.multivariate_normal(mu_q, cov_q,p)
    dataP = np.concatenate((dataP, newP))
    dataQ = np.concatenate((dataQ, newQ))
    allDataP[i] = dataP
    allDataQ[i] = dataQ

    
    
