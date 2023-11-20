from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os
from bs4 import BeautifulSoup

SCROLL_PAUSE_SEC = 3

def scroll_down():
    global driver
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SEC)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            time.sleep(SCROLL_PAUSE_SEC)
            new_height = driver.execute_script("return document.body.scrollHeight")

            try:
                driver.find_element_by_class_name("mye4qd").click()
            except:

               if new_height == last_height:
                   break


        last_height = new_height

# keywords = ['happy person', 'sad face person', 'surprised person', 'angry person']
keywords = ['sad acting to cry on']

for keyword in keywords:
    url = 'https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjgwPKzqtXuAhWW62EKHRjtBvcQ_AUoAXoECBEQAw&biw=768&bih=712'.format(keyword)

    driver = webdriver.Chrome()
    driver.get(url)

    time.sleep(1)

    scroll_down()

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    images = soup.find_all('img', attrs={'class':'rg_i Q4LuWd'})

    print('number of img tags: ', len(images))

    n = 1
    for i in images:

        try:
            imgUrl = i["src"]
        except:
            imgUrl = i["data-src"]
            
        with urllib.request.urlopen(imgUrl) as f:
            with open('/Users/hongdagyeong/Documents/Pd/project_face_classification/mini_project/img_data/' + keyword + '/' + keyword + '_' + str(n) + '.jpg', 'wb') as h:
                img = f.read()
                h.write(img)

        n += 1