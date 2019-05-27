# -*- coding: utf-8 -*- 
import os 
import matplotlib.pyplot as plt
import numpy as np

path = 'D:/NM02001/Desktop/3.14-01_zhao.txt'
picname = path.split('/')[-1]

tem=[]
th=[]
i = 0
file = open(path,'r')
data = file.readlines()
for line in data:
    if i>3:
        ob = line.split()
        tem.append(float(ob[0]))
        th.append(float(ob[1]
        ))
    i=i+1

x = np.array(tem)
y = np.array(th)

plt.plot(x, y)
plt.xlabel('tem')
plt.ylabel('th')
plt.title(picname)
plt.show()