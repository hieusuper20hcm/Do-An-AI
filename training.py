import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def getImagesAndLabels(path):
    # Lấy tất cả các file trong thư mục
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    IDs=[]

    # Lặp qua tất cả các đường dẫn hình ảnh và tải các id và hình ảnh
    for imagePath in imagePaths:
        # tải hình ảnh và chuyển đổi nó sang thang màu xám
        faceImg=Image.open(imagePath).convert('L');

        # Bây giờ chúng tôi đang chuyển đổi hình ảnh PIL thành mảng numpy
        faceNp=np.array(faceImg,'uint8')

        # lấy Id từ hình ảnh
        ID=int(os.path.split(imagePath)[-1].split('.')[1])

        # trích xuất khuôn mặt từ mẫu hình ảnh đào tạo
        faces.append(faceNp)

        # Thêm khuôn mặt đó vào danh sách cũng như Id của khuôn mặt đó
        IDs.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

# Lấy các khuôn mặt và ID từ thư mục dataset
Ids,faces=getImagesAndLabels(path)

# Train model để trích xuất đặc trưng các khuôn mặt và gán với từng tên
recognizer.train(faces,Ids)
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

# Lưu model
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()