import os
import tensorflow as tf
import numpy as np
import dataSet
import time
import math
import random
#保持随机数字每次都一样
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)
#定义变量
batch_size = 32
classes = ['dogs','cats']
num_classes = len(classes)

validation_size = 0.2#测试数据所占比例
img_size = 64#图片尺寸
num_channels = 3#通道数
train_path = '/Users/liang-yulong/Desktop/dataset_kaggledogvscat/'#数据路径
#读取数据，获得数据类型Datasets，
data = dataSet.read_train_set(train_path,img_size,classes,validation_size)
print("reading input_data")
#定义模型结构 3层卷积层 2层全连接层
sess = tf.Session()
x = tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')
y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name="y_true")
y_true_cls = tf.argmax(y_true,dimension=1)
#卷积核为3*3，卷积数为32
filter_size_conv1 = 3
num_filters_conv1 =32
#卷积核为3*3，卷积数为64
filter_size_conv2 = 3
num_filters_conv2 =64
#卷积核为3*3，卷积数为64
filter_size_conv3 = 3
num_filters_conv3 =64
#全连接层，尺寸为1024
fc_layer_size = 1024
#定义变量weight、biases的创建函数，节省代码数量
def create_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def create_biases(size):
    return tf.Variable(tf.constant(0.05,shape=[size]))
#定义卷积层，参数为输入层、通道数、卷积核大小、卷积数
def create_convolution_layer(input,num_input_channels,conv_filter_size,num_filter):
    weights = create_weight(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filter])
    biases = create_biases(num_filter)
    layer = tf.nn.conv2d(input,weights,[1,1,1,1],padding="SAME")
    layer+=biases
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],"SAME")
    return layer
#创建卷积层与全连接层的转换函数
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer,[-1,num_features])
    return layer
#创建全连接层
def create_fc_layer(input,num_inputs,num_outputs,use_relu = True):
    weight = create_weight(shape=[num_inputs,num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input,weight)+biases
    layer = tf.nn.dropout(layer,keep_prob=0.7)
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
#创建输出函数，用于提示进程
def show_progress(epoch,feed_dict_train,feed_dict_validate,val_loss,i):
    acc = sess.run(accuracy,feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy,feed_dict=feed_dict_validate)
    print("epoch:",str(epoch+1)+",i:",str(i)+",acc:",str(acc)+",val_acc:",str(val_acc)+",val_loss:",str(val_loss))

#models
layer_conv1 = create_convolution_layer(x,num_channels,filter_size_conv1,num_filters_conv1)
layer_conv2 = create_convolution_layer(layer_conv1,num_filters_conv1,filter_size_conv2,num_filters_conv2)
layer_conv3 = create_convolution_layer(layer_conv2,num_filters_conv2,filter_size_conv3,num_filters_conv3)
layer_flat = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(input=layer_flat,num_inputs=layer_flat.get_shape()[1:4].num_elements(),num_outputs=fc_layer_size,use_relu=True)
layer_fc2 = create_fc_layer(input=layer_fc1,num_inputs=fc_layer_size,num_outputs=num_classes,use_relu=False)
#run train
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.arg_max(y_pred,dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels = y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())

total_iterations = 0
saver = tf.train.Saver()

def train(num_iterations):
    global total_iterations
    for i in range(total_iterations,total_iterations+num_iterations):
        x_batch,y_true_batch,_,cls_batch = data.train.next_batch(batch_size)
        x_vali_batch,y_vali_batch,_,cls_vali_batch = data.valid.next_batch(batch_size)
        feed_dict_train = {x:x_batch,y_true:y_true_batch}
        feed_dict_Vali = {x:x_vali_batch,y_true:y_vali_batch}

        sess.run(optimizer,feed_dict=feed_dict_train)
        example = data.train.num_examples()
        if i% int(example/batch_size) == 0:
            val_loss = sess.run(cost,feed_dict=feed_dict_Vali)
            epoch = int(i/int(example/batch_size))

            show_progress(epoch,feed_dict_train,feed_dict_Vali,val_loss,i)
            saver.save(sess,'./2/model/dog_cat.ckpt',global_step=i)
    total_iterations+=num_iterations

train(num_iterations=8000)