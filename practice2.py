
# coding: utf-8

# In[195]:


current_x=0.1
learnling_rate=0.1
num_iterations=100
def slope_at_given_x_value(x):
    return 5*x**4-6*x**2
for i in range(num_iterations):
    previous_x=current_x
    current_x+= learnling_rate*slope_at_given_x_value(previous_x)
    print(previous_x)
print("The local minimum occurs at %f" % current_x)
print(slope_at_given_x_value(current_x))


# #  梯度下降法来求最低点的x值
# - 可以发现随着训练的迭代，x在逐渐接近函数最小值对应的y值
# - 最后输出的是梯度，可以发现已经非常接近0了
# - 下面更改一下学习率的大小，迭代的次数再来跑一遍

# In[202]:


current_x=0.1
learnling_rate=0.3
num_iterations=100
def slope_at_given_x_value(x):
    return 5*x**4-6*x**2
for i in range(num_iterations):
    previous_x=current_x
    current_x+= -learnling_rate*slope_at_given_x_value(previous_x)
    print(previous_x)
print("The local minimum occurs at %f" % current_x)
print(slope_at_given_x_value(current_x))


# - 改了学习率和迭代次数，可以发现梯度更小了。
