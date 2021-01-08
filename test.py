import cv2
import sqlite3
from PIL import Image
import pickle
cam = cv2.VideoCapture(0, cv2.FONT_HERSHEY_SIMPLEX)
detector=cv2.CascadeClassifier('D:\\NhanDien\\haarcascade_frontalface_default.xml')

# Hàm cập nhật tên và ID vào CSDL
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("D:\Download\SQLiteStudio\FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name='"+str(Name)+"' WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(ID, Name) VALUES("+str(Id)+",'"+str(Name)+"')"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
id=input('enter your id ')
name=input('enter your name ')
insertOrUpdate(id,name)
sampleNum=0
while(True):
    #camera read
    ret, img = cam.read() 

    # Lật ảnh cho đỡ bị ngược   
    img = cv2.flip(img,1)


    # Chuyển ảnh về xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh camera
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    # Đăt số thứ tự cho hình ở ngoài dataset
        sampleNum=sampleNum+1

          # Ghi dữ liệu khuôn mặt vào thư mục dataSet
        cv2.imwrite("dataset/user."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>100:
        break
cam.release()
cv2.destroyAllWindows()