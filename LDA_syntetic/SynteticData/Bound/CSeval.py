import numpy as np
from utils import *
import matplotlib.pyplot as plt
repeatTime=1
ds=range(2,100)
a=[]
b=[]
c=[]
for d in ds:
    for _ in range(repeatTime):
        #d = 10
        X = np.random.random_sample(size=(d,d))
        X = np.matrix((X + X.T)/2)
        
        x = np.matrix(np.random.random_sample(size=(d,1)))
        
        a.append(norm(x)*normOperator(X))
        b.append(norm(X*x))
        c.append((np.sqrt(np.matrix.trace(X*X)).getA()[0][0])*norm(x))
        
plt.plot(ds,b,label="norm(X*x)")
plt.plot(ds,a,label="norm(x)*normOperator(X)")
plt.plot(ds,c,label="sqrt(trace(X*X))*norm(x)")
plt.legend().draggable()
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.show()