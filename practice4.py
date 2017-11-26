
# coding: utf-8

# In[149]:



from numpy import array
import numpy as np
from random import choice
num_iterations=1000
w=np.random.rand(3)-0.5
learning_rata=0.01
f=lambda x :0 if x< 0  else 1
training_data = [ (array([0,0,1]), 0), #第三个相当与参数b
                    (array([0,1,1]), 1), 
                    (array([1,0,1]), 1), 
                    (array([1,1,1]), 1) ] 
for i in range (num_iterations):
    x, truth =choice(training_data)#这个地方不能用random.choice不知道为什么
    result=np.dot(x,w)
    error=truth-f(result)
    w+= learning_rata*error*x#这个地方必须是加，不能是减不知道为什么。（将参数b当成一个输入，并且权值为1）因为第三个特征是一直是1，所以不用将w和b分开更新
testing_data= [ (array([0,0,1]), 0), 
                    (array([1,1,1]), 1),
                    (array([1,0,1]), 1), 
                    (array([1,1,1]), 1),
                   (array([0,1,1]), 1), 
                  (array([0,1,1]), 1),
                   (array([1,0,1]), 1),         
                 (array([0,0,1]), 0), 
                    (array([1,0,1]), 1),
                 (array([0,0,1]), 0), 
                    (array([0,1,1]), 1),
                    (array([1,1,1]), 1)]    
for i in range(len(testing_data))  :
    x,_=choice(testing_data)
    result=np.dot(x,w)
    print("{}:{}->{}".format( x[:2],result,f(result)))
    


# ## 一个简单的感知器，使用逻辑“或”来实现，
# - 还有几个问题，没想明白，更新参数的时候不知道为什么使用减不行
# - choice的问题
