# coding:utf-8
import threading
import time
from collections import namedtuple
from concurrent import futures
import requests
import csv
import codecs

header = ['aid', 'view', 'danmaku', 'reply', 'favorite', 'coin', 'share']
Video = namedtuple('Video', header)
headers = {
    'X-Requesred-With': 'XMLHttpRequest',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/56.0.2924.87 Safari/537.36'
}
total = 1
result = []
lock = threading.Lock()

def run(url):
    global total
    req = requests.get(url, headers = headers, timeout = 6).json()
    time.sleep(0.5)
    try:
        data = req['data']
        video = Video(
            data['aid'],
            data['view'],
            data['danmaku'],
            data['reply'],
            data['favorite'],
            data['coin'],
            data['share']
        )
        with lock:
            result.append(video)
            print(total)
            total += 1
    except:
        pass

def save():
    with codecs.open('D:\Program Files\Python\\result.csv', 'w+', 'utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(result)

if __name__ == '__main__':
    urls = ['http://api.bilibili.com/archive_stat/stat?aid={}'.format(i) for i in range(100000)]
    with futures.ThreadPoolExecutor(32) as executor:
        executor.map(run, urls)
    save()
    print 'spider finished'