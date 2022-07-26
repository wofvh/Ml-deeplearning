from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus


baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' # 네이버 검색url
plusUrl = input('강호동,이만기 : ') # 검색어 질문
url = baseUrl + quote_plus(plusUrl) # url로 이동하기위한 쿼리문자열 만들기
html = urlopen(url) # url 열기
soup = bs(html, "html.parser")
img = soup.find_all(class_='_img', limit = 50)