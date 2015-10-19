from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.lda import LDA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import copy
from utils import *

k=1
d=3
T = 0.5
clfWindowSize = 350
initLen = clfWindowSize
violationThreshold = k-1


digits = datasets.load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
X = []
y = []
binaryLabels = []
pca = PCA(n_components=d)
data = pca.fit_transform(data, labels)
dataPerDigit=[]
labelsPerDigits=[]
allP=[]
allQ=[]
for i in range(2):
    indecies = np.where(labels==i)[0]
    currDigit = data[indecies].tolist()
    dataPerDigit.append(currDigit)
    X = X + currDigit
    if i%2==0:
        allP = allP + currDigit
        currLabels = np.zeros(len(currDigit)).tolist()
    else:
        allQ = allQ + currDigit
        currLabels = np.ones(len(currDigit)).tolist()
    y = y + currLabels
    labelsPerDigits.append(currLabels)
print X
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LDA()
clf.fit(a_train, b_train)

print metrics.accuracy_score(clf.predict(a_test), b_test)
