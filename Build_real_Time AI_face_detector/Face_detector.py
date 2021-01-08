import cv2
from random import randrange
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img1=cv2.imread('appu.jpg')
# img=cv2.resize(img1,(400,500))
webcam=cv2.VideoCapture(0)
while True:
    SR,frame=webcam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinate=trained_face_data.detectMultiScale(gray)
# x,y,w,h=face_coordinate[0]

    for (x,y,w,h) in face_coordinate:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),4)
     # print(face_coordinate)
    cv2.imshow('hello',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()