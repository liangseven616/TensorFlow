from cv2 import cv2
import os
import numpy as np
import random

def create_dir(*args):
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMG_size = 64

def getpaddingsize(shape):
    h,w = shape
    longest = max(h,w)
    result = (np.array([longest*4,int])-np.array([h,h,w,w],int))//2
    return result.tolist()

def dealwithImage(img,h=64,w = 64):
    top,bottom,left,right = getpaddingsize(img.shape[0:2])
    img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
    img = cv2.resize(img,(h,w))
    return img

def relight(imgsrc,alpha = 1,bias = 0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc*alpha + bias
    imgsrc[imgsrc<0] = 0
    imgsrc[imgsrc>255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getFaceFromCamera(outdir):
    create_dir(outdir)
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('/Users/liang-yulong/Desktop/FaceAndEyeDetect/haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        if (n <= 50):
            print('It`s processing %s image.' % n)
            # 读帧
            success,img = camera.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (IMG_size, IMG_size))
                #could deal with face to train
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                cv2.putText(img, 'haha', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('please input yourename: ')
    getFaceFromCamera(os.path.join('/Users/liang-yulong/Desktop/face', name))


