# coding=utf-8
'''
Created on 2017年9月20日

@author: pigfish
'''

from baike_spider import url_manager, html_downloader, html_parser, html_outputer
import urllib

class spiderMain(object):
    def __init__(self):
        self.urls = url_manager.UrlManager()
        self.downloader = html_downloader.HtmlDownloader()
        self.parser = html_parser.HtmlParser()
        self.outputer = html_outputer.HtmlOutputer()
    
    def craw(self, root_url):
        # 设置爬取网页计数器，至多爬取1000个
        count = 1
        # 根网页开始爬取
        self.urls.addNewUrl(root_url)
        while self.urls.hasNewUrl():
            try:
                newUrl = self.urls.getNewUrl()
                print('craw %d : %s' %(count, urllib.unquote(str(newUrl))))
                htmlCont = self.downloader.download(newUrl)
                newUrls, newCont = self.parser.parse(newUrl,htmlCont)
                self.urls.addNewUrls(newUrls)
                self.outputer.collectData(newCont)
            except:
                print('craw failed')       
            if count == 100:
                break
            else:
                count += 1          
        self.outputer.outputHtml()
            
            
if __name__ == '__main__':
    root_url = 'https://lvyou.baidu.com/notes/'
    objSpider = spiderMain()
    objSpider.craw(root_url)