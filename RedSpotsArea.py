import numpy as np
import cv2
import sys

path=sys.argv[1]

image = cv2.imread(path)
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([22, 93, 0], dtype="uint8")
upper = np.array([45, 255, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area=[]
area_to_sort=[]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area.append(w*h)
    area_to_sort.append(w*h)
    #cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
 
a=area.index(max(area))

b=area_to_sort

b=b.sort()

a2=area.index(area_to_sort[-2])

x,y,w,h = cv2.boundingRect(cnts[a])   

x2,y2,w2,h2 = cv2.boundingRect(cnts[a2])   

if w>h:
    image = cv2.imread(path)
    image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area=[]
    area_to_sort=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area.append(w*h)
        area_to_sort.append(w*h)
    #cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
 
    a=area.index(max(area))

    b=area_to_sort

    b=b.sort()

    a2=area.index(area_to_sort[-2])

    x,y,w,h = cv2.boundingRect(cnts[a])   
    
    x2,y2,w2,h2 = cv2.boundingRect(cnts[a2])

croppedplate=original[(y):(y+h),(x):(x+w)]
cv2.imwrite('platecropped.jpg',croppedplate)



croppedplate=cv2.imread('platecropped.jpg')

gray = cv2.cvtColor(croppedplate, cv2.COLOR_BGR2GRAY)

# compute the Scharr gradient magnitude reprtesentation of the images 
# in both the x and y direction 
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

#substract the y-gradient from the x-gradient 
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

#blur and threshold the image 
blurred = cv2.blur(gradient, (9,9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations 
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
 
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(croppedplate, [box], -1, (0, 255, 0), 2)
cv2.imwrite('br.jpg',croppedplate)


without_yellow_edges=original[(y+w2):(y+h-w2),(x+w2):(x+w-w2)]
cv2.imwrite('without_yellow_edges.jpg',without_yellow_edges)

without_yellow_edges1=croppedplate[w2:h-w2,w2:w-w2]
cv2.imwrite('without_yellow_edges1.jpg',without_yellow_edges1)



wye=cv2.imread('without_yellow_edges.jpg')
wye1 = wye.copy()
wye2 = wye.copy()
wye = cv2.cvtColor(wye, cv2.COLOR_BGR2HSV)
lower = np.array([22, 93, 0], dtype="uint8")
upper = np.array([45, 255, 255], dtype="uint8")
mask = cv2.inRange(wye, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imwrite('yellowcircles.jpg',wye)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area=[]
area_to_sort=[]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area.append(w*h)
    area_to_sort.append(w*h)
    cv2.rectangle(wye1, (x, y), (x + w, y + h), (36,255,12), 2)
    
aa=area_to_sort.sort()

circlerects=[]

barwelldist=[]

for i in range(1,5):
    indx=area.index(area_to_sort[-i])
    area[indx]=0
    xx,yy,ww,hh=cv2.boundingRect(cnts[indx])
    circlerects.append([xx,yy,ww,hh])
    d=math.sqrt((box[0][0]-xx)*(box[0][0]-xx)+(box[0][1]-yy)*(box[0][1]-yy))
    barwelldist.append(d)

sortbwd=np.argsort(barwelldist)
j=1
area_covered=[]
for i in sortbwd:
    xx,yy,ww,hh=circlerects[i]
    wn='Well'+str(j)+'.jpg'
    wncrop=wye2[(yy):(yy+hh),(xx):(xx+ww)]
    cv2.imwrite(wn,wncrop)
    wcopy=wncrop.copy()
    red_channel = wcopy[:,:,2]
    wcopy[np.where(red_channel<90)]=0
    l=0
    for k in np.where(red_channel<90):
        l=l+len(k)
    cv2.imwrite('redregion'+str(j)+'.jpg',wcopy)
    a=(l/(wcopy.shape[0]*wcopy.shape[1]))*100
    area_covered.append(a)
    print("Area Covered By Red Spot in Well "+str(j)+" is "+str(a)+"%")
    j=j+1