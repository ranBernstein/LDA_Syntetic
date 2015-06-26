import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *

mu_p_0 = np.array([[2],[2]])
mu_q_0 = np.array([[-2],[-2]])
cov_p_0 = np.array([[1,1],\
                    [1,2]])
cov_q_0 = np.array([[2,-1],\
                    [-1,1]])

k=100
T=0.5
n_p = 50
n_q = n_p
n = n_p + n_q
beforSyncLength = 100
timeLength=10000
dataLength = beforSyncLength + timeLength + n_p
#Data before last sync
Xp =  np.random.multivariate_normal(mu_p_0.T[0], cov_p_0, dataLength)
#Xp = np.matrix(Xp)
Xq =  np.random.multivariate_normal(mu_q_0.T[0], cov_q_0, dataLength)
#Xq = np.matrix(Xq)



Xp_0 =  Xp[:beforSyncLength]
Xq_0 =  Xq[:beforSyncLength]

Xp =  Xp[n_p:]
Xq =  Xq[n_p:]

#S0, x0, y0, w0, w0_norm, B0_inverted = calcWindowParams(Xp_0, Xq_0)

x0, y0 = mu_p_0, mu_q_0
S0 = cov_p_0+cov_q_0 + x0*x0.T + y0*y0.T
w0, B0_inverted = FLDformula(x0, y0, S0)
w0_norm = norm(w0)

index = 0
cosines = []
#allDataP = np.array(np.reshape(Xp[0], (1,2)))
#allDataQ = np.array(np.reshape(Xq[0], (1,2)))

allDataP = np.array(Xp[:2])
allDataQ = np.array(Xq[:2])


nodeData = []
for i in range(k):
    #referenceParams = (S0, x0, y0, w0, w0_norm, B0_inverted)
    nodeData.append((np.reshape(Xp[0], (1,2)), np.reshape(Xq[0], (1,2))))
violations = [] 
S_old, x_old, y_old, w_old, w_old_norm, B_old_inverted = (S0, x0, y0, w0, w0_norm, B0_inverted)
while index < timeLength:
    roundViolations = 0.0
    for i in range(k):
        dataP, dataQ = nodeData[i]
        #S_old, x_old, y_old, w_old, w_old_norm, B_old_inverted \
        #    = calcWindowParams(oldDataP, oldDataQ)
        
        #newDataP = np.concatenate((oldDataP[1:] ,np.reshape(Xp[i], (1,2))))
        #newDataQ = np.concatenate((oldDataQ[1:] ,np.reshape(Xq[i], (1,2))))
        
        dataP = np.concatenate((dataP ,np.reshape(Xp[index], (1,2))))
        dataQ = np.concatenate((dataQ ,np.reshape(Xq[index], (1,2))))
        
        nodeData[i] = (dataP, dataQ)
        isInSafeZone = checkLocalConstraint(S_old, x_old, y_old, w_old_norm, \
            B_old_inverted, dataP, dataQ, T)
        if isInSafeZone:
            roundViolations += 1
        
        allDataP =  np.concatenate((allDataP ,np.reshape(Xp[index], (1,2))))
        allDataQ = np.concatenate((allDataQ ,np.reshape(Xq[index], (1,2))))
        S, x, y, w, w_norm, B_inverted = calcWindowParams(allDataP, allDataQ) 
        c = cosineSimilarity(w0, w)      
        cosines.append(c)

        index += 1
    violations.append(roundViolations/k)
dic={}
dic['k'] = k
dic['T'] = T
plt.plot(cosines)
plt.title('Cosine similarity of the whole data'+str(dic))
plt.xlabel('Number of samples of the whole data (Time)')
plt.ylabel('Cosine similarity with w0')
plt.figure()
plt.plot(violations)
plt.xlabel("Number of observations that have been accumulated in the node (epochs)")
plt.ylabel('Fraction of nodes that are in the safe zone')
plt.title('Fraction of nodes in safe zone'+str(dic))
plt.show()
    
    
    
    
    
    