import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np

#length, width, type
data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, .5, 1],
        [2, .5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

myst = [4.5, 1]


#params
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()


def NN(width, length):
  
  
  return
# NN



def sigmoid(x):
  return 1/(1+np.exp(-x))
# sigmoid

def sigmoid_p(x):
  return sigmoid(x) * (1-sigmoid(x))
# sigmoid_p

# X = np.linspace(-6, 6, 50)
# plt.plot(X, sigmoid(X), c='r')
# plt.plot(X, sigmoid_p(X), c='b')
# plt.savefig('test.png')

#data scatter

# plt.axis([0, 6, 0, 6])
# plt.grid()
# for i in range(len(data)):
  # point = data[i]
  # color = 'r'
  # if point[2] == 0:
    # color = 'b'
  # plt.scatter(point[0], point[1], c=color)
# plt.savefig('data.png')


learning_rate = 0.2
costs = []

#training loop
for i in range(50000):
  ri = np.random.randint(len(data))
  point = data[ri]
  
  z = point[0]*w1 + point[1] * w2 + b
  pred = sigmoid(z)
  
  target = point[2]
  cost = np.square(pred - target)
  
  dcost_dpred = 2 * (pred - target)
  dpred_dz = sigmoid_p(z)
  dcost_dz = dcost_dpred * dpred_dz
  
  dz_dw1 = point[0]
  dz_dw2 = point[1]
  dz_db = 1
  
  dcost_dw1 = dcost_dz * dz_dw1
  dcost_dw2 = dcost_dz * dz_dw1
  dcost_db = dcost_db * dz_db
  
  w1 = w1 - learning_rate * dcost_dw1
  w2 = w2 - learning_rate * dcost_dw2
  b = b - learning_rate * dcost_db
  
  if i % 100 == 0:
    cost_sum = 0
    for j in range(len(data)):
      p = data[j]
      z = p[0]*w1 + p[1] * w2 + b
      pred = sigmoid(z)
      cost_sum += np.square(pred - target)
    costs.append(cost_sum/len(data))
  
#

plt.plot(costs)
plt.savefig('costs.png')