
# coding: utf-8

# In[161]:



import numpy as np
import pandas as pd#用pandas读取数据
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
 
class Perceptron(object):
    
    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,X,y):
        
        self.w_ = np.zeros(1 + X.shape[1]) # add w_0　　　　　#初始化权重。数据集特征维数+1。
        self.errors_ = []#用于记录每一轮中误分类的样本数
         
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))#调用了predict()函数
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
     
    def net_input(self,X):
       
        return np.dot(X,self.w_[1:]) + self.w_[0]#计算向量点乘
     
    def predict(self,X):#预测类别标记
        
        return np.where(self.net_input(X) >= 0.0,1,-1)
    
    
    
    
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)#读取数据还可以用request这个包
#header=None时，即指明原始文件数据没有列索引，这样read_csv为自动加上列索引，除非你给定列索引的名字。
print(df.tail())#输出最后五行数据，看一下Iris数据集格式

"""抽取出前100条样本，这正好是Setosa和Versicolor对应的样本，我们将Versicolor
对应的数据作为类别1，Setosa对应的作为-1。对于特征，我们抽取出sepal length和petal
length两维度特征，然后用散点图对数据进行可视化"""  
 
y = df.iloc[0:100,4].values  #一种索引方式

y = np.where(y == 'Iris-setosa',-1,1)

X = df.iloc[0:100,[0,2]].values


plt.scatter(X[:50,0],X[:50,1],color = 'red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal lenght')
plt.legend(loc='upper left')
plt.show()
 
#train our perceptron model now
#为了更好地了解感知机训练过程，我们将每一轮的误分类
#数目可视化出来，检查算法是否收敛和找到分界线
ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epoches')
plt.ylabel('Number of misclassifications')
plt.show()
 
#画分界线超平面
def plot_decision_region(X,y,classifier,resolution=0.02):
    #setup marker generator and color map
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
     
    #plot the desicion surface
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1              
     
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
     
    plt.contour(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
     
    #plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx), marker=markers[idx],label=cl)
 
plot_decision_region(X,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upperleft')
plt.show()

