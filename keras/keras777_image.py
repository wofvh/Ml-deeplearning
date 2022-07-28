import cv2
import numpy as np
import os


path_dir = "D:\downloads\\asian_round_face_women"
file_list = os.listdir(path_dir)

#print(file_list)

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))

print(file_name_list)

#1장으로 테스트해본거. 잘됨

# image = cv2.imread('img_14.jpg')
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
#     cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
#     cropped = image[y: y+h, x: x+w]
#     resize = cv2.resize(cropped, (250,250))
#     cv2.imshow("crop&resize", resize)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    


def Cutting_face_save(image, name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (250,250))
        # cv2.imshow("crop&resize", resize)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 이미지 저장하기
        cv2.imwrite(f"D:\downloads\\asian_round_face_women/{name}.jpg", resize)
        
for name in file_name_list:
    img = cv2.imread("D:\downloads\\asian_round_face_women/"+name+".jpg")
    Cutting_face_save(img, name)