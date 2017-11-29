
# coding: utf-8

# In[2]:


import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import os
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('mnist_data',one_hot=True)
sess = tf.InteractiveSession()

def imageprocess(filename):#图像预处理函数
    im=Image.open(filename)
    out = im.resize((28,28),Image.ANTIALIAS)
    i=out.convert("L")

    im_array = np.array(i)
    im_mean=np.mean(im_array)
    im_array1=im_array-im_mean#防止在真的纸上手写的数字的时候纸板的其他位置有污垢影响图像的像素的问题，将有些灰黑色的黑影或者是污垢重新处理成白色。
    for i in range(28):
        for j in range(28):
            if (im_array1[i,j]<0):
                im_array1[i,j]+=im_mean
            else:
                im_array1[i,j]=255
    plt.imshow(im_array1, cmap = plt.cm.gray)
    plt.show()
    im_array2=1-im_array1*(1/255)
    im_array3=im_array2.reshape((1,784))
    #im_y=np.array([0,0,1,0,0,0,0,0,0,0],float)
    #im_y
    return im_array3


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
for i in range(10000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})#没有返回值

saver.save(sess, 'C:/Users/lenovo/Desktop/canshu/ca/model.ckpt')

print("训练结束")
print("官方测试集：")
sum_rate=0
for i in range(100):
    test_batch=mnist.test.next_batch(100)
    train_accuracy=accuracy.eval(feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})#又返回值
    sum_rate+=train_accuracy
print ("training accuracy %g"%(sum_rate/100))    
truth=[]
yy=[]
file_path="C:\\Users\\lenovo\\Desktop\\test"

dirs=os.listdir(file_path)
for dir in dirs:
    string_number=dir[0]
    number=int(string_number)
    print(number)
    truth.append(number)
    image_path=os.path.join(file_path,dir)
    result=imageprocess(image_path)
    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: result,keep_prob: 1.0}, session=sess)
    yy.append(predint[0])
    print('预测结果:{}'.format(predint))

    print("---------------------------------------------------------------")
correct_prediction = tf.equal(truth, yy)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("测试集准确率为：{}".format(accuracy))





# In[1]:


file_path="C:\\Users\\lenovo\\Desktop\\test"

dirs=os.listdir(file_path)
for dir in dirs:
    string_number=dir[0]
    number=int(string_number)
    print(number)
    truth.append(number)
    image_path=os.path.join(file_path,dir)
    result=imageprocess(image_path)
    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: result,keep_prob: 1.0}, session=sess)
    yy.append(predint[0])
    print('预测结果:{}'.format(predint))

    print("---------------------------------------------------------------")
correct_prediction = tf.equal(truth, yy)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("测试集准确率为：{}".format(accuracy))


# In[7]:


import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('mnist_data',one_hot=True)
test_data = mnist.test.images
test_data
def display(i):
    img = test_data[i]
  
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()
for i in range (10):
    display(i)
tf.argmax(test_data,1)    
#test_data=test_data[1].reshape(28,28)
#test_data


# > 
#   (1)tf.argmax(input, axis=None, name=None, dimension=None) 
#     此函数是对矩阵按行或列计算最大值 
#     参数 
# 
#         input：输入Tensor 
#         axis：0表示按列，1表示按行 
#         name：名称 
#         dimension：和axis功能一样，默认axis取值优先。新加的字段 
# 
#     返回：Tensor 行或列的最大值下标向量 
# 
#     (2)tf.equal(a, b) 
#     此函数比较等维度的a, b矩阵相应位置的元素是否相等，相等返回True,否则为False 
#     返回：同维度的矩阵，元素值为True或False 
# 
#     (3)tf.cast(x, dtype, name=None) 
#     将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 
#     那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以 
# 
#     (4)tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None) 
#      功能：求某维度的最大值 
#     (5)tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None) 
#     功能：求某维度的均值 
# 
#     参数1--input_tensor:待求值的tensor。 
#     参数2--reduction_indices:在哪一维上求解。0表示按列，1表示按行 
#     参数（3）（4）可忽略 
#     例：x = [ 1, 2 
#               3, 4] 
#     x = tf.constant([[1,2],[3,4]], "float") 
#     tf.reduce_mean(x) = 2.5 
#     tf.reduce_mean(x, 0) = [2, 3] 
#     tf.reduce_mean(x, 1) = [1.5, 3.5] 
# 
#     (6)tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
#     从截断的正态分布中输出随机值 
# 
#         shape: 输出的张量的维度尺寸。 
#         mean: 正态分布的均值。 
#         stddev: 正态分布的标准差。 
#         dtype: 输出的类型。 
#         seed: 一个整数，当设置之后，每次生成的随机数都一样。 
#         name: 操作的名字。 
# 
#     （7）tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
#     从标准正态分布中输出随机值 
# 
#     (8) tf.nn.conv2d(input, filter, strides, padding,  
#                      use_cudnn_on_gpu=None, data_format=None, name=None) 
#     在给定的4D input与 filter下计算2D卷积 
#         1，输入shape为 [batch, height, width, in_channels]: batch为图片数量，in_channels为图片通道数 
#         2，第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width,  
#             in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数， 
#             卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input 
#             的第四维 
#         3，第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4 
#         4，第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍） 
#         5，第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true 
# 
#         结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。 
# 
#     (9)tf.nn.max_pool(value, ksize, strides, padding, name=None) 
#     参数是四个，和卷积很类似： 
#     第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map， 
#         依然是[batch, height, width, channels]这样的shape 
#     第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们 
#         不想在batch和channels上做池化，所以这两个维度设为了1 
#     第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1] 
#     第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME' 
#     返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式 
# 
#     (10) tf.reshape(tensor, shape, name=None) 
#     函数的作用是将tensor变换为参数shape的形式。 
#     其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小， 
#     函数会自动计算，但列表中只能存在一个-1。（当然如果存在多个-1，就是一个存在多解的方程了） 
# 
#     (11)tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None)  
#     为了减少过拟合，随机扔掉一些神经元，这些神经元不参与权重的更新和运算 
#     参数： 
#         x            :  输入tensor 
#         keep_prob    :  float类型，每个元素被保留下来的概率 
#         noise_shape  : 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。 
#         seed         : 整形变量，随机数种子。 
#         name         : 名字，没啥用。  
#     '''  

# ## 下面是根据训练保存的参数，重新提取出来应用在重新搭建的和原来一样的模型上，来进行测试。
# - 好像全关闭之后，第一次可以运行，后面就是经常显示参数缺失，还有问题待解决

# In[ ]:


from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def imageprocess(filename):
    im=Image.open(filename)
    out = im.resize((28,28),Image.ANTIALIAS)
    i=out.convert("L")

    im_array = np.array(i)
    im_mean=np.mean(im_array)
    im_array1=im_array-im_mean
    for i in range(28):
        for j in range(28):
            if (im_array1[i,j]<0):
                im_array1[i,j]+=im_mean
            else:
                im_array1[i,j]=255
    plt.imshow(im_array1, cmap = plt.cm.gray)
    plt.show()
    im_array2=1-im_array1*(1/255)
    im_array3=im_array2.reshape((1,784))
    
    
    return im_array3



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
file_path="C:\\Users\\lenovo\\Desktop\\test"

with tf.Session() as sess:
    sess.run(init_op)
    print("1111111111")
    print ("Model restoring.")
    model_data = tf.train.latest_checkpoint('C:/Users/lenovo/Desktop/canshu/ca/')  
    saver.restore(sess, model_data) 
    #saver.restore(sess, "C:/Users/lenovo/Desktop/canshu/ca/model.ckpt")#这里使用了之前保存的模型参数
    truth=[]
    yy=[]
    dirs=os.listdir(file_path)
    for dir in dirs:
        string_number=dir[0]
        number=int(string_number)
        print(number)
        truth.append(number)
        image_path=os.path.join(file_path,dir)
        result=imageprocess(image_path)
        prediction=tf.argmax(y_conv,1)
        predint=prediction.eval(feed_dict={x: result,keep_prob: 1.0}, session=sess)
        yy.append(predint[0])
        print('预测结果:{}'.format(predint))
       
        print("---------------------------------------------------------------")
    correct_prediction = tf.equal(truth, yy)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("测试集准确率为：{}".format(accuracy))

