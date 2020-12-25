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
    gons=np.array([[(210,height-40),(500,height-40),(220,110)]],'int64')
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,gons,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def display(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image
	

    


cap=cv2.VideoCapture("test2.mp4")

while(cap.isOpened()):
    _,frame=cap.read()
    b = cv2.resize(frame, (640, 480)) 
    
    canny_image=canny(b)

    cropped=region_of_interest(canny_image)

    line=cv2.HoughLinesP(cropped,2,np.pi/180,80,np.array([]),minLineLength=10,maxLineGap=2)

    line_image=display(b,line)
	
    combo_image=cv2.addWeighted(b,0.8,line_image,1,1)#line image
    cv2.imshow("re",combo_image)
  
    if cv2.waitKey(1)== ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
     
  