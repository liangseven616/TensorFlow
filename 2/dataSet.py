import tensorflow as tf
import numpy as np
import os
import glob
import cv2
from cv2 import cv2
from sklearn.utils import shuffle

class Dataset(object):
    #数据类型,用于存储数据images,labels,img_names,cls
    def __init__(self,images,labels,img_names,cls):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0
    def image(self):
        return self._images
    def labels(self):        
        return self._labels    
    def img_names(self):        
        return self._img_names    
    def cls(self):        
        return self._cls    
    def num_examples(self):        
        return self._num_examples    
    def epochs_done(self):        
        return self._epochs_done    
    def next_batch(self,batch_size):        
        start=self._index_in_epoch        
        self._index_in_epoch+=batch_size

        if self._index_in_epoch>self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size<=self._num_examples
        end = self._index_in_epoch
        return self._images[start:end],self._labels[start:end],self._img_names[start:end],self._cls[start:end]


def load_train(train_path,image_size,classes):
    #读取train_path下的文件，获得数据；image_size为图片大小，classes为图片类型数组
    images = []
    labels = []
    img_names = []
    cls = []
    print('going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (indexs:{})'.format(fields,index))
        path = train_path + classes[index]
        files = os.listdir(path)
        for f in files:
            if f.split('.')[-1] == 'jpg': 
                f = os.path.join(path,f)
                image = cv2.imread(f)
                image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image,1.0/255.0)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                f_base = os.path.basename(f)
                img_names.append(f_base)
                cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images,labels,img_names,cls

def read_train_set(train_path,image_size,classes,validation_size):
    class Datasets(object):
        pass
    datasets = Datasets()

    images,labels,img_names,cls = load_train(train_path,image_size,classes)
    images, labels, img_names, cls = shuffle(images,labels,img_names,cls)
    #按照比例分配数据，分为训练集和验证集
    if isinstance(validation_size,float):
        validation_size = int(validation_size*images.shape[0])
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    datasets.train = Dataset(train_images,train_labels,train_img_names,train_cls)
    datasets.valid = Dataset(validation_images, validation_labels, validation_img_names, validation_cls)
    return datasets