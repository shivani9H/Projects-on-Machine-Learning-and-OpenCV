import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
    imgBlur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(imgBlur,50,150)
    return canny

def region_of_interest(image):
    height=image.shape[0]
    gons=np.array([[(210,height-40),(500,height-40),(220,110)]],'int32')
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,gons,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

    def make_coordinate(image,line_parameter):
    slope,intercept=line_parameter
#     print(image.shape)
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
def average_slope_intercept(image,lines):
    left_f=[]
    right_f=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameter=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameter[0]
        intercept=parameter[1]
        if slope<0:
            left_f.append((slope,intercept))
        else:
            right_f.append((slope,intercept))
            
    left_f_average=np.average(left_f,axis=0)
    right_f_average=np.average(right_f,axis=0)
    left_line=make_coordinate(image,left_f_average)
    right_line=make_coordinate(image,right_f_average)

    return np.array([left_line,right_line])
    
   













image=cv2.imread('test_image.jpg')
gray=np.copy(image)
canny_image=canny(gray)
# plt.imshow(canny_image)
# plt.show()
cv2.imshow('vcv',image)
crop=region_of_interest(canny_image)
cv2.imshow('gray',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()