from cv2 import cv2
import numpy as np 
from PIL import Image
import pytesseract as pt

img = cv2.imread("resources\Text_1.jpg")

text = pt.image_to_string(img)
print(text)

cv2.imshow('Image', img)
cv2.waitKey(0)