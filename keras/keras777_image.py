from ast import keyword
from urllib import response
from matplotlib.pyplot import ginput
# pip install git+https://github.com/Joeclinton1/google-images-download.git
from google_images_download import google_images_download
from sympy import limit

def googleImageCrawling(keyword, limit):
    response = response = google_images_download.googleimagesdownload()  

arguments = {"keywords":keyword,"limit": limit,
             "print_urls": True,"chromedriver": "./chromedriver", "fromat":"jpg"}   
paths = response.download(arguments)  
print(paths)  

keyword = input("키워드로 입력해주세요(복수 선정 시 ',' 붙여서 입력해주세요!): ")
limit = input("이미지 개수를 입력해주세요 : " )

googleImageCrawling(keyword, int(limit))
