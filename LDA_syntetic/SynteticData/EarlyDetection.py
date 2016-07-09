import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from random import sample
from utils import *
import copy
from sklearn.lda import LDA
font = {
    'family': 'normal',
    'weight': 'normal',
    'size': 18
    }
import matplotlib
matplotlib.rc('font', **font)
d=2
R=1.0
var=0.001
T=0.5
k=2
L=4000
dic={}
mu_p_0 = np.zeros((d,))
mu_q_0 = np.zeros((d,))
theta = np.pi/4
mu_q_0[0] = R*np.cos(theta)
mu_q_0[1] = R*np.sin(theta)
mu_p_0[0] = -R*np.cos(theta)
mu_p_0[1] = -R*np.sin(theta)
cov_p_0 = var*np.diag(np.ones((d,)))
cov_q_0 = cov_p_0
mu_p = copy.copy(mu_p_0)
mu_q = copy.copy(mu_q_0)
cov_p = copy.copy(cov_p_0)
cov_q = copy.copy(cov_q_0)
cosines = []
cosinesPer = []
time=0
allDataP, allDataQ, references, allP, allQ = initNodesData(k,L,d,mu_p_0,mu_q_0,cov_p_0,cov_q_0)
S0, x0, y0, w0, w0_norm, B0 = calcWindowParams2D(allDataP, allDataQ)
u0 = x0-y0
S0_per, x0_per, y0_per, w0_per, w0_norm_per, \
    B0_per = calcWindowParams2D(allDataP, allDataQ)
u0_per = x0_per - y0_per


W=1
clf = LDA()   
X = allP + allQ
y= np.ones(len(allP)).tolist()+(-1*np.ones(len(allQ))).tolist()
clf.fit(X,y)
timeLength=16
params =  range(timeLength)
first = True
notFoundViolation = True
for time in params:
    R0 = getR0(w0_norm, T)

    
    X=[]
    y=[]
    Xp, Xq = [], []
    for i in range(k):
        newP = np.random.multivariate_normal(mu_p, cov_p,1)
        X.append(newP[0])
        y.append(1)
        Xp.append(newP[0])
        newQ = np.random.multivariate_normal(mu_q, cov_q,1)
        X.append(newQ[0])
        y.append(-1)
        Xq.append(newQ[0])
        allDataP[i] = np.concatenate((allDataP[i][1:], newP))        
        allDataQ[i] = np.concatenate((allDataQ[i][1:], newQ))
        currentData=allDataP[i], allDataQ[i]
        globalParams = w0, B0
        v,_ = getLeftSide(references[i],  
                globalParams, currentData, R0, alpha=1)
        print "v", v/R0
        if(v/R0 > 1) and notFoundViolation:
            firstViolatedX = newP
            notFoundViolation = False
    #X = newP.tolist() + newQ.tolist()
    #y= np.ones().tolist()+(-1*np.ones()).tolist()
    print "score", clf.score(X,y)
    #clf.fit(X,y)
    
    w = clf.coef_
    a = w[1]/w[0]
    xx = np.linspace(-2, 2)
    yy = a * xx #- (clf.intercept_[0]) / w[1]
    #plt.figure()
    Xp, Xq=np.array(Xp), np.array(Xq)
    #plt.scatter(X[:,0], X[:,1], c='b')
    if first:
        plt.plot(xx, yy, 'k-', label="Classifier")
        first=False
    plt.scatter(Xp[:,0],Xp[:,1], s=150, c='b', marker='_')
    plt.scatter(Xq[:,0],Xq[:,1], s=150, c='r', marker='+')
    theta += np.pi/30
    mu_q[0] = R*np.cos(theta)
    mu_q[1] = R*np.sin(theta)
    mu_p[0] = -R*np.cos(theta)
    mu_p[1] = -R*np.sin(theta)
t = plt.text(-1.1, -0.7, "$t_{I}$", ha="center", va="center", rotation=0,
            size=20)

#t = plt.text(1, 0.7, "t1", ha="center", va="center", rotation=0,size=14)
t = plt.text(1.1, -0.7, "$t_{L}$", ha="center", va="center", rotation=0,
            size=20)


t = plt.text(1.1, 0.7, "$t_{I}$", ha="center", va="center", rotation=0,
            size=20)

#t = plt.text(1, 0.7, "t1", ha="center", va="center", rotation=0,size=14)
t = plt.text(-1.1, 0.7, "$t_{L}$", ha="center", va="center", rotation=0,
            size=20)
#t = plt.text(-1.2, 0.7, "t_last-1", ha="center", va="center", rotation=0,size=14)

t = plt.text(1.5, 1, "+", ha="center", va="center", rotation=0, color='r',
            size=80)
t = plt.text(-1.5, -1, "_", ha="center", va="center", rotation=0, color='b',
            size=80)

#plt.annotate("First Violation",size=14, xy=(-0.3, -1), xytext=(-0.3, -2),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
plt.annotate("First Violation",size=14, xy=firstViolatedX[0], xytext=(-0.3, -2),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.legend().draggable()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()




