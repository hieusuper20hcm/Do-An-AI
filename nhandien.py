import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('D:\\NhanDien\\haarcascade_frontalface_default.xml')

image = cv2.imread("dataset/user.3.1.jpg")

# Bước 2: Tạo một bức ảnh xám
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces=faceDetect.detectMultiScale(grayImage,1.3,5);

# Bước 4: Vẽ các khuôn mặt đã nhận diện được lên tấm ảnh gốc
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
