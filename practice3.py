
# coding: utf-8

# In[227]:


wheat_and_bread=[[0.5,5],[0.6,5.5],[0.8,6],[1.1,6.8],[1.4,7]]
def step_gradient(b,m,points,learningrate):
    b_gridient=0
    m_gridient=0
    n=float(len(points))
    for i in range(0,len(points)):
        x=points[i][0]
        y=points[i][1]
        b_gridient+= -(2/n)*(y-(m*x+b))#这里直接按推到出来的梯度公式写的梯度，没有计算误差的和
        m_gridient+= -(2/n)*x*(y-(m*x+b))
        new_b=b-learningrate*b_gridient
        new_m=m-learningrate*m_gridient
    return [new_b,new_m]
def gratient_descent(num_iterations,points,starting_b,starting_m,learningrate):
    b=starting_b
    m=starting_m
    for i in range(0,num_iterations):
        previous_b=b
        previous_m=m
        b,m=step_gradient(previous_b,previous_m,points,learningrate)
        print(previous_b,previous_m)
        
    return [b,m]
b,m=gratient_descent(100,wheat_and_bread,1,1,0.1)
print("last b:{} and m:{}".format(b,m))
print("开始函数式子：y=x+1")
print("训练完的式子：y={}*x+{}".format(m,b))
print("训练完的式子：y=%d*x+%d"%(m,b))


# ## 还有需要改进的地方，下一步加上最后运算结果的二维直线图。
# - 这里几乎没有使用库函数，比如求导部分没有，二是直接用的求导结果的公式。
# - 前面两个练习都是针对每个算法给出的代码，跟线性回归没关系，最小二乘法只计算了损失的均值，梯度下降只算了一元函数的自变量x的减小，而不是这里的任何参数。
# 
# 
