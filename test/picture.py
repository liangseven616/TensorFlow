import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,10,50,90,97])
y1 = np.array([0,31.6,53.8,84.7,98.1])
y2 = np.array([0,28.02,53.56,91.15,112.2])
y3 = y2-y1
z1 = np.polyfit(x, y1, 3) # 用3次多项式拟合
z2 = np.polyfit(x, y2, 3) # 用3次多项式拟合
z3 = np.polyfit(x, y3, 3) # 用3次多项式拟合

p1 = np.poly1d(z1)
p2 = np.poly1d(z2)
p3 = np.poly1d(z3)
print(p1,"\n",p2,"\n",p3) # 在屏幕上打印拟合多项式
yvals1=p1(x) # 也可以使用yvals=np.polyval(z1,x)
yvals2=p2(x) # 也可以使用yvals=np.polyval(z1,x)
yvals3=p3(x) # 也可以使用yvals=np.polyval(z1,x)

plot1=plt.plot(x, yvals1, 'r',label='y1')
plot2=plt.plot(x, yvals2, 'b',label='y2')
plot3=plt.plot(x, yvals3, 'g',label='y2-y1')


plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()
plt.savefig('p1.png')