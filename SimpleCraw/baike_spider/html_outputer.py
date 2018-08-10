# coding:utf8
'''
Created on 2017年9月20日

@author: pigfish
'''

import urllib

class HtmlOutputer(object):
    
    def __init__(self):
        self.datas = []
    
    def collectData(self,data):
        if data is None:
            return 
        else:
            self.datas.append(data)

    
    def outputHtml(self):
        with open('output.html','w') as outPut:
            outPut.write('<html>')
            outPut.write('<head><meta charset="utf-8"></head>')
            outPut.write('<body>')
            outPut.write('<table>')
            
            for data in self.datas:
                outPut.write('<tr>')
                url = urllib.unquote(str(data['url']))
                outPut.write('<td>%s</td>' % url)
                outPut.write('<td>%s</td>' % data['title'].encode('utf-8'))
                outPut.write('<td>%s</td>' % data['summary'].encode('utf-8'))
                outPut.write('</tr>')
            
            outPut.write('</table>')
            outPut.write('</body>')
            outPut.write('</html>')
    
    
    
    



