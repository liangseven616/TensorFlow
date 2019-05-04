from hyperlpr import *
import cv2.cv2 as cv2

image = cv2.imread("C:/Users/Liang/Pictures/demo.png")
print(HyperLPR_PlateRecogntion(image))
cv2.imshow("src",image)