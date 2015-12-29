import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from utils import *
dic={}
k=200
p_tp = 0.94
p_fp = 0.875
p_tv = 0.195
#l=1000

values = range(k+1)
print 
B_tp = binom.pmf(values, k, p_tp)
B_fp = binom.pmf(values, k, p_fp)
#P_tv = np.random.binomial(1, p_tv, l)

res = B_tp*p_tv/(B_tp*p_tv + B_fp*(1-p_tv))
#y, x = np.histogram(res)
#modValue
sigma = np.std(res)
dic['k'] = k
dic['p_tp'] = p_tp
dic['p_fp'] = p_fp
dic['p_tv'] = p_tv

#plt.hist(res,50, weights=np.zeros_like(res) + 1. / res.size)
plt.plot(values, res)
plt.title(dic)

plt.figure()
plt.title(dic)

mu = p_fp*k
dis = np.array(values) - mu
res = chernof(k,p_fp,dis)
plt.plot(values, res)
plt.show()