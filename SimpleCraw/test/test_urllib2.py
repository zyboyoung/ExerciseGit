# coding:utf8
'''
Created on 2017年9月15日

@author: pigfish
'''

import urllib2
import cookielib

url = "http://www.baidu.com"

# urllib2下载网页的第一种方法，最简洁
print 'the first method'
response1 = urllib2.urlopen(url)
print response1.getcode()
print len(response1.read())
 
# urllib2下载网页的第二种方法，添加data、http header
print 'the second method'
request = urllib2.Request(url)
request.add_header('user-agent','Mozilla/5.0')
response2 = urllib2.urlopen(request)
print response2.getcode()
print len(response2.read())
 
# urllib2下载网页的第三种方法，添加特殊情景的处理器
print 'the third method'
cj = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
urllib2.install_opener(opener)
response3 = urllib2.urlopen(url)
print response3.getcode()
print cj
print response3.read()