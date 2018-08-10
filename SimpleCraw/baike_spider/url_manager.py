# coding:utf-8
'''
Created on 2017年9月20日

@author: pigfish
'''


# url管理器需要有两个列表，一个待爬取，一个已爬取
class UrlManager(object):
    def __init__(self):
        self.newUrls = set()
        self.oldUrls = set()
        
    
    # 向管理器中添加一个新的url
    def addNewUrl(self,url):
        if url is None:
            return
        if url not in self.newUrls and url not in self.oldUrls:
            self.newUrls.add(url)

    # 从管理器中获取一个新的url
    def getNewUrl(self):
        url = self.newUrls.pop()
        self.oldUrls.add(url)
        return url 

    # 判断管理器中是否有新的待爬取url
    def hasNewUrl(self):
        if len(self.newUrls) != 0:
            return 1
        else:
            return 0

    # 向管理器中添加批量的url
    def addNewUrls(self,urls):
        if urls is None or len(urls)==0:
            return
        for url in urls:
            self.addNewUrl(url)
    
    
    
    
    
    
    
    



