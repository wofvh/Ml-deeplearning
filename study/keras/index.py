#!/usr/bin/env python3
# Anchor extraction from HTML document
from bs4 import BeautifulSoup
from urllib.request import urlopen
response = urlopen('https://www.naver.com/')
soup = BeautifulSoup(response, 'html.parser')
for anchor in soup.soup.select("span.ah_k"):
    print(anchor)