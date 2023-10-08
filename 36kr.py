import pytz
from bs4 import BeautifulSoup
import datetime
import ssl
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
url1=[]


ssl._create_default_https_context = ssl._create_unverified_context
tz = pytz.timezone('Asia/Shanghai')

url = 'https://36kr.com/newsflashes'
#driver.set_window_size(500,500)

#匹配所有在description" content="后面的并以"结尾
aa=r'"description" content="(.*?)"'

#爬取36kr

aa=r'"description" content="(.*?)"'
headers={'Accept-Encoding':'gzip, deflate, br','User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36'}
html = requests.get(url,headers=headers)
html=html.text
soup = BeautifulSoup(html, 'html.parser')
ps=soup.find_all('a', attrs={'class': 'item-title'})

for i in ps:
    url_all="https://36kr.com"+i.get("href")
    html = requests.get(url_all,headers=headers)
    #print(html.text)
    tt=re.findall(aa,html.text)

    print(tt)
time.sleep(3)



















def GetWeekday():
    weekday = datetime.datetime.now(tz).weekday()
    if weekday == 0:
        return '星期一'
    elif weekday == 1:
        return '星期二'
    elif weekday == 2:
        return '星期三'
    elif weekday == 3:
        return '星期四'
    elif weekday == 4:
        return '星期五'
    elif weekday == 5:
        return '星期六'
    elif weekday == 6:
        return '星期日'


'''if __name__ == '__main__':
    index = 1
    ps = getDocument(html, 'a', {'class': 'item-title'})
    date = datetime.datetime.now(tz)
    # print(str(date.year) + '年' + str(date.month) + '月' + str(date.day) + '日' + GetWeekday() + ',' + '每日科技快讯：')
    text = str(date.year) + '年' + str(date.month) + '月' + str(date.day) + '日' + GetWeekday() + ',' + '每日科技快讯：' + '\\n> '
    for p in ps:
        text += '##### ' + str(index) + '.' + p.text + ' \\n> '
        # print(text)
        index += 1
    print(text)'''
    

'''
XPATH
//*[@id="app"]/div/div[2]/div[3]/div/div/div/div[2]/div[1]/div[2]/pre
'''
