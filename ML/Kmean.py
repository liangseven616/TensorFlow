from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np 

data = make_blobs(n_samples=500,centers=5,random_state=8)
x,y = data

clf = KNeighborsClassifier()
clf.fit(x,y)
print('分类是：',clf.predict([[6,-10]]))
print('correct:',clf.score(x,y))

plt.scatter(6,-10,marker='*',c='red',s=200)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')
plt.show()

