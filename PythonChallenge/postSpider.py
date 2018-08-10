import requests
from xml.parsers.expat import ParserCreate

class DefaultSaxHandler(object):
    def __init__(self, provinces):
        self.provinces = provinces

    # 处理标签开始
    def start_element(self, name, attrs):
        if name != 'map':
            name = attrs['title']
            number = attrs['href']
            self.provinces.append((name, number))

    # 处理标签结束
    def end_element(self, name):
        pass

    # 文本处理
    def char_data(self, text):
        pass

def get_provinces_entry(url):

    # 记住解码
    contents = requests.get(url).content.decode('gb2312')
    start = contents.find('<map name="map_86" id="map_86">')
    end = contents.find('</map>')

    # 切片
    content = contents[start:end + len('</map>')].strip()
    provinces = []

    # 生成Sax处理器
    handler = DefaultSaxHandler(provinces)

    # 创建返回一个 xmlparser 对象
    parser = ParserCreate()
    parser.StartElementHandler = handler.start_element
    parser.EndElementHandler = handler.end_element
    parser.CharacterDataHandler = handler.char_data

    # 解析数据
    parser.Parse(content)

    return provinces

if __name__ == '__main__':
    provinces = get_provinces_entry('http://www.ip138.com/post')
    print(provinces)

