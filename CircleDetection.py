#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:47:09 2021

@author: abc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:22:25 2021

@author: abc
"""
import sys
import numpy as np
import cv2
import math

image = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# equalize lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# edge enhancement
edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U,ksize = 3, scale = 1, delta = 0)
#cv2.imwrite("Edges", edge_enh)
#cv2.waitKey(0)
#retval = cv2.imwrite("edge_enh.jpg", edge_enh)

# bilateral blur, which keeps edges
blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)

# use simple thresholding. adaptive thresholding might be more robust
(_, thresh) = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
#cv2.imshow("Thresholded", thresh)
#cv2.waitKey(0)
#retval = cv2.imwrite("thresh.jpg", thresh)

# do some morphology to isolate just the barcode blob
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
#cv2.imshow("After morphology", closed)
#cv2.waitKey(0)
#retval = cv2.imwrite("closed.jpg", closed)

# find contours left in the image
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
print(box)
#cv2.imshow("found barcode", image)
#cv2.waitKey(0)
#retval = cv2.imwrite("found.jpg", image)

BarCode_x=(box[1][0]+((box[0][0]+box[1][0])/2))/2

Circle_centre = BarCode_x
Circle_y=box[2][1]
circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param2=40,minRadius=0,maxRadius=100)

distance=[]

for i in range(len(circles[0])):
#    center_coordinates = (circles[0][i][0], circles[0][i][1])
#    radius = circles[0][i][2]
#    color = (255, 0, 0)
#    thickness = 2
#    image_circle = cv2.circle(image, center_coordinates, radius, color, thickness)
#    cv2.imwrite('Cv2HoughCircles.jpg',image_circle)
    distance.append(abs(Circle_centre-circles[0][i][0]))

#distance.sort()
s=sorted(range(len(distance)), key=lambda k: distance[k])   
#nearest_well_index=distance.index(min(distance))  

indices = [i for i, x in enumerate(distance) if x < 10]
y_dist=[]
if len(indices) > 0:
    for i in indices:
        if (Circle_y-circles[0][i][1])>0:
            y_dist.append(math.sqrt((circles[0][i][1]-Circle_y)*(circles[0][i][1]-Circle_y)+(circles[0][i][0]-Circle_centre)*(circles[0][i][0]-Circle_centre)))
        else:
            y_dist.append(100000)

ii=y_dist.index(min(y_dist)) 
 
nearest_well_index=indices[ii]

i=nearest_well_index
  
center_coordinates = (circles[0][i][0], circles[0][i][1])
radius = circles[0][i][2]
color = (255, 0, 0)
thickness = 2
image_circle = cv2.circle(image, center_coordinates, radius, color, thickness)
cv2.imwrite('Cv2HoughCircles.jpg',image_circle) 

#dist_from_nearest=y_dist[ii]
  
#max_dist_from_well=250

