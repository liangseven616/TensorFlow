import os
import tensorflow as tf 
import numpy as np
import time
from cv2 import cv2
from scipy import io

DEFAULT_PATH = "/Users/liang-yulong/Library/Containers/com.apple.Safari/Data/Desktop/imagenet-vgg-verydeep-19.mat"
VGG19_Layers = ('conv1_1','relu1_1','conv1_2','relu1_2','pool1',
                'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
                'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
                'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
                'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5',              
                'fc6','relu6',              
                'fc7','relu7',              
                'fc8','softmax')

#input shape：[batch,height,width,channels] 只有一张图，所以batch = 1

INPUT_SHAPE = [1,224,224,3]

class VGG:
    def __init__(self,model_path = None):
        print("Load Pre-Trained Model")
        mat = None
        if model_path == None:
            mat = io.loadmat(DEFAULT_PATH)
        else:
            mat = io.loadmat(model_path)
        assert mat != None
        norm_pic = mat['normalization'][0][0][0]
        print(norm_pic)
        self.mean_pix = np.mean(norm_pic,axis=(0,1))
        self.layer_param = mat['layers'][0]

    def build_VGGnet(self):
        print('building VGG net now!')
        self.image = tf.placeholder(tf.float32,shape=INPUT_SHAPE,name='input_image')
        self.layer = {}
        last_layer = self.image

        assert last_layer != None
        
        for i,name in enumerate(VGG19_Layers):
            types = name[:3]
            temp_layer = None
            if types == 'con':
                filters,bias = self.layer_param[i][0][0][0][0]
                filters = np.transpose(filters,(1,0,2,3))
                bias = np.reshape(bias,-1)
                conv = tf.nn.conv2d(last_layer,tf.constant(filters),strides=[1,1,1,1],padding='SAME')
                temp_layer = tf.nn.bias_add(conv,tf.constant(bias))
            elif types == 'rel':
                temp_layer = tf.nn.relu(last_layer)
            elif types == 'poo':
                temp_layer = tf.nn.max_pool(last_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            elif name == 'fc6':                
                weights,bias = self.layer_param[i][0][0][0][0]                
                bias = np.reshape(bias,-1)                
                #flatten the output from pooling layer                
                last_layer = tf.reshape(last_layer,[last_layer.shape[0],-1])                
                weights = np.reshape(weights,(-1,weights.shape[-1]))                
                temp_layer = tf.nn.bias_add(tf.matmul(last_layer,weights),bias)            
            elif name == 'fc7':                
                weights,bias = self.layer_param[i][0][0][0][0]                
                bias = np.reshape(bias,-1)                
                weights = np.reshape(weights,(-1,weights.shape[-1]))                
                temp_layer = tf.nn.bias_add(tf.matmul(last_layer,weights),bias)            
            elif name == 'fc8':                
                weights,bias = self.layer_param[i][0][0][0][0]                
                bias = np.reshape(bias,-1)                
                weights = np.reshape(weights,(-1,weights.shape[-1]))                
                temp_layer = tf.nn.bias_add(tf.matmul(last_layer,weights),bias)           
            elif name == 'softmax':                
                temp_layer = tf.nn.softmax(last_layer) 
            
            assert temp_layer !=None
            self.layer[name] = temp_layer
            last_layer = temp_layer

    def predict(self,image_path = None):
        if image_path is None:
            image_path = "/Users/liang-yulong/Library/Containers/com.apple.Safari/Data/Desktop/timg.jpg"

        labels = [str.strip() for str in open("/Users/liang-yulong/Desktop/TensorFlow/3/synset.txt").readlines()]
        img = cv2.imread(image_path)
        assert img is not None
        img = img[:,:,::-1]
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float)
        img = np.expand_dims(img,axis=0)
        img = img - self.mean_pix

        with tf.Session() as sess:
            image_feed = {self.image:img}
            prob = self.layer['softmax'][0].eval(feed_dict = image_feed)
            
            maxIndex = np.argmax(prob)
            print("index:",maxIndex)
            print("prob:",prob[maxIndex])
            print("label:",labels[maxIndex])

if __name__ == "__main__":
    vgg = VGG()
    vgg.build_VGGnet()
    vgg.predict('/Users/liang-yulong/Desktop/timg.jpg')

