import cv2
import numpy as np

image = cv2.imread('best 1.jpg')

img=image

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

#cv2.imwrite('shadows_out.png', result)
#cv2.imwrite('shadows_out_norm.png', result_norm)

image=result_norm

start_point = (1800, 750) 

end_point = (2000, 900)

color = (255, 0, 0) 

image = cv2.rectangle(image, start_point, end_point, color, 2) 

cv2.imwrite('bestwell.jpg',image)

crop_img=image[750:900, 1800:2000]

cv2.imwrite('cropwell.jpg',crop_img)


minDist = 100
param1 = 30 #500
param2 = 100 #200 #smaller value-> more false circles
minRadius = 1
maxRadius = 1000 #10

# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)


detected_circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2 = 30, minRadius = 1, maxRadius = 40) 

center_coordinates = (1900, 840)
radius = 50
color = (255, 0, 0)
thickness = 2
image_circle = cv2.circle(image, center_coordinates, radius, color, thickness)

cv2.imwrite('image_circle.jpg',image_circle)
