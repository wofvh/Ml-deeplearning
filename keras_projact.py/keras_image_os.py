import os
import glob
import pandas

#파일 이름 변경하기

file_path = 'C:/study/_data/test/leejehoon/'
# file_path = 'D:/study_data/_project1/img/fashion_img/paris_val/'
file_names = os.listdir(file_path)
file_names

# file_names = glob.glob('D:/study_data/_project1/img/2019FW/london/2019FW_train/*.jpg')
# file_names.sort(key=os.path.getctime, reverse=False)
# import natsort
# file_names = natsort.natsorted(file_names, reverse=False)


i = 1
# for f in file_names:
#     print(f)
#     src = os.path.join(file_names, f)
#     dst = str(i) + '.jpg'
#     dst = os.path.join(f, dst)
#     os.rename(src, dst)
#     i += 1

for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
    