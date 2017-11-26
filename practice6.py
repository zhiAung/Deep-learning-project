
# coding: utf-8

# In[277]:


import numpy as np
import pandas as pd
x=np.array([[0,0,1],
          [1,0,1],
          [0,1,1],
          [1,1,1]])
y=np.array([[0],[1],[1],[0]])
num_iterations=60000
np.random.seed(1)
w1=2*np.random.rand(3,4)-1
w2=2*np.random.rand(4,1)-1
#

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

for _ in range(num_iterations):
    a1=np.dot(x,w1)
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)
    z2=sigmoid(a2)
    error=z2-y
    delta2=error*sigmoid_derivative(z2)#注意这里中间不再是矩阵的乘法，而是对位的简单相乘。这里直接用的导数公式，中间步奏
    delta1=delta2.dot(w2.T)*sigmoid_derivative(z1)#同理
    d_w2=(z1.T).dot(delta2)#这里如何确定相乘的顺序，以及是否需要转置，第一 画图，写出来各个矩阵的维数，第二检查矩阵中各个参数的值是如何得出的，
    #比如这里w的每一个分向量应该由x1,x2,x3,x4来公共更新，所以，z1需要转置。
    d_w1=x.T.dot(delta1)
    w2-=d_w2
    w1-=d_w1

func=lambda x:0 if x<0.3 else 1


xx=x.tolist()#将二维矩阵转为一维列表
z3=z2.flatten()#将二维矩阵转为一维数组

z_list=[]
for i in range(len(z3)):
    z_list.append(func(z3[i]))
print(xx)    
print("{}->{}".format(z3,z_list))    
    


# ## 神经网络实现 
# - 用两层的简单神经网络实现“异或”逻辑
# - 需要注意的
#   1. 各个矩阵的维数
#   2. 矩阵乘法前后的顺序，以及是否需要转置
#   3. 矩阵乘和普通乘
