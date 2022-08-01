import os
from glob import glob
import pandas as pd

label_list = os.listdir('D:/test/') # test내의 폴더들 for문에 넣기 위해 list형식으로 정의
# label_list = os.listdir('D:\test\choiminsik/') # test내의 폴더들 for문에 넣기 위해 list형식으로 정의

image_paths = [] # image_path들을 받아넣을 list정의

for label in label_list:      # 위에 정의된 label_list를 label로 하나씩 넣어 줌
    temp = glob(f'D:/test/{label}/*.jpg') # for 문이 돌며 해당 경로 폴더 내에 있는 jpg파일들을 list 덩어리로 크게 가져옴
    
    for image_path in temp: # list를 다시 풀어서 각 각 하나의 인자(string형식) 으로 위에 정의된 image_paths에 입력
        image_paths.append(image_path) 

data = pd.DataFrame(image_paths)
data.to_csv('D:/study_data/test.csv')


from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
from urllib import request
import requests

print('인공지능 얼굴인식 관상 프로그램')

facial_features_cordinates = {}


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])
                                                                                                                                                                                                    

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    print(facial_features_cordinates)
    return output


def face_func(url):

    # 사용자 입력 받기
    # url = input('관상을 보고 싶은 image의 URL을 입력하시오 : ')

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    #url = "https://cdn.fastpick.co.kr/fastpick/2021/03/image_2322861011616642090118.png"
    # res = request.urlopen(url).read()
    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(res)

    # URL 이미지 보여주기
    image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)
        output = visualize_facial_landmarks(image, shape)
        cv2.imshow("Image", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()