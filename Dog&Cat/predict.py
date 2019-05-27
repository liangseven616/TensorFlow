import tensorflow as tf 
import numpy as np
import os
import sys
from cv2 import cv2

image_size=64
num_channels=3
images=[]

path = "C:/Users/Liang/Pictures/2.jfif"
image = cv2.imread(path)
print("address:",path)
image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
images.append(image)
images = np.array(images,dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images,1.0/255.0)

for img in images:
    x_batch = img.reshape(1,image_size,image_size,num_channels)
    sess = tf.Session()

    saver = tf.train.import_meta_graph('./2/model/dog_cat.ckpt-7500.meta')
    saver.restore(sess,'./2/model/dog_cat.ckpt-7500')

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_image = np.zeros((1,2))
    feed_dict_testing = {x:x_batch,y_true:y_test_image}
    result = sess.run(y_pred,feed_dict_testing)

    res_label = ['dog','cat']
    print(res_label[result.argmax()])
