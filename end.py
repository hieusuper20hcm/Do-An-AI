import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3

faceDetect=cv2.CascadeClassifier('D:\\NhanDien\\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0, cv2.FONT_HERSHEY_SIMPLEX)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainningData.yml")

#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203,23,252)

#get data from sqlite by ID
def getProfile(id):
    conn=sqlite3.connect("D:\Download\SQLiteStudio\FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

face_names = []
while(True):
    #camera read
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(
           gray, 1.1,5
    )
    face_names = []
    for(x,y,w,h) in faces:
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        name="Unknow"
        if(conf<80):
            profile=getProfile(id)
            name=str(profile[1])        
            face_names.append(name)
            for(x,y,w,h),name in zip(faces,face_names):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,name, (x+10,y+h+30), fontface, fontscale, fontcolor ,2)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,name, (x+10,y+h+30), fontface, fontscale, fontcolor ,2)

    cv2.imshow('Face',img) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
