import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3
import face_recognition

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0, cv2.FONT_HERSHEY_SIMPLEX)

# Khởi tạo bộ nhận diện khuôn mặt
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")

#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203,23,252)

# Hàm lấy thông tin người dùng qua ID
def getProfile(id):
    conn=sqlite3.connect("D:\Download\SQLiteStudio\FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

while(True):
    # Đọc ảnh từ camera
    ret,img=cam.read()

    # Lật ảnh cho đỡ bị ngược
    img = cv2.flip(img, 1)

    # Chuyển ảnh về xám
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh camera
    faces = faceDetect.detectMultiScale(
           gray, 1.3,5
    )


    for(x,y,w,h) in faces:
          # Vẽ hình chữ nhật quanh mặt
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        # Nhận diện khuôn mặt, trả ra 2 tham số id: mã nhân viên và dist (dộ sai khác)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        profile=None

        # Nếu độ sai khác < 40% thì lấy profile
        if(conf<=40):
            profile=getProfile(id)

        # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
        if profile!=None:
            name=str(profile[1])     
            cv2.putText(img,name, (x+10,y+h+30), fontface, fontscale, fontcolor ,2)   
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"Unknow", (x+10,y+h+30), fontface, fontscale, fontcolor ,2)

    cv2.imshow('Face',img) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
