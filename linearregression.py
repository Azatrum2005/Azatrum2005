import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('data.txt',delimiter=',')
x=data[:,0]
y=data[:,1]
y=y.reshape(y.size,1)
x=np.vstack((np.ones((x.size,)),x)).T
print(x,y)
#plt.scatter(x[:,1],y)
#plt.show()
def model(x,y,iteration,learningrate):
    global theta,cost_list
    m = y.size
    theta = np.zeros((2, 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(x, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred - y))
        d_theta = (1/m)*np.dot(x.T, y_pred - y)
        theta = theta - learningrate*d_theta
        cost_list.append(cost)
    return theta, cost_list
model(x,y,100000,0.000000005)
print(np.dot([1,1600], theta),cost_list[99999])
rng = np.arange(0, 100000)
plt.plot(rng,cost_list)
plt.show()

'''from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
xs=np.array ([1,2,3,4,5,6], dtype=np.float64)
ys=np.array ([5,4,6,5,6,7], dtype=np.float64)
def best_fit_slope_and_intercept(xs,ys):
    m=(((mean (xs)*mean (ys)) -mean (xs*ys)) / ((mean (xs) *mean(xs))-mean (xs*xs)))
    b=mean(ys)-m*mean (xs) 
    return m, b

m,b=best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()'''
