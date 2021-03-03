#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:20:09 2021

@author: abc
"""

import cv2
import numpy as np

image = cv2.imread('IMG_6023.JPG')

lower_black = np.array([0,0, 0], dtype=np.uint8)
upper_black = np.array([0,0,0], dtype=np.uint8)

mask = cv2.inRange(image, lower_black, upper_black)
output = cv2.bitwise_and(image, image, mask = mask)

cv2.imwrite("barcodeblack.jpg", np.hstack([image, output]))

image = cv2.imread('IMG_6023.JPG')

start_point=(850, 3040)

end_point = (1130, 3600)

color = (255, 0, 0) 

img = cv2.rectangle(image, start_point, end_point, color, 2) 

cv2.imwrite("barcoderectangle.jpg", img)

bimg = image[3040:3600,850:1130]

cv2.imwrite('BarCodeCropImage.jpg',bimg)
