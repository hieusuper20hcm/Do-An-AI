import cv2
import sqlite3
from PIL import Image
import pickle
cam = cv2.VideoCapture(0, cv2.FONT_HERSHEY_SIMPLEX)
detector=cv2.CascadeClassifier('D:\\NhanDien\\haarcascade_frontalface_default.xml')

#insert/update data to sqlite
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataset/user."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()