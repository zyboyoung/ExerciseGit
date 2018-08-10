# coding=utf-8
from bs4 import BeautifulSoup
import re
import urlparse


# 需要解析出新的url列表以及网页内容
class HtmlParser(object):
    

    def getNewUrls(self, pageUrl, soup):
        newUrls = set()
        links = soup.find_all('a',href=re.compile(r'/notes/*'))
        for link in links:
            newUrl = link['href']         
            newFullUrl = urlparse.urljoin(pageUrl, newUrl)
            newUrls.add(newFullUrl)
        
        return newUrls
    
    def getNewData(self, pageUrl, soup):
        # data为字典类型
        data = {}
        data['url'] = pageUrl
        
        # <dd class="lemmaWgt-lemmaTitle-title"><h1>胶水语言</h1>
        titleCont = soup.find('dd',class_='lemmaWgt-lemmaTitle-title').find('h1')
        data['title'] = titleCont.get_text()
        
        # <div class="lemma-summary" label-module="lemmaSummary">
        summaryCont = soup.find('div',class_='lemma-summary')
        data['summary'] = summaryCont.get_text()
        
        return data
    
    def parse(self,pageUrl,htmlCont):
        if pageUrl is None or htmlCont is None:
            return None,None
        else:
            soup = BeautifulSoup(htmlCont,'html.parser',
                                 from_encoding='utf-8')
            newUrls = self.getNewUrls(pageUrl,soup)
            newData = self.getNewData(pageUrl,soup)
            return newUrls,newData

    
    



