import bz2
# import requests
# import re

# url = 'http://www.pythonchallenge.com/pc/def/integrity.html'
# response = requests.get(url)
# html_file = response.text
# re_pattern = re.compile(r'<!--[\d\D]*-->')
# html_data = re_pattern.findall(html_file)


# data = str(html_data).split('\\n')
# un = data[1].split('\'')[1]
# pw = data[2].split('\'')[1]
# print(un)
# print(pw)

# un = bytes(un, 'ascii')
# pw = bytes(pw, 'utf-8')


un = b'BZh91AY&SYA\xaf\x82\r\x00\x00\x01\x01\x80\x02\xc0\x02\x00 \x00!\x9ah3M\x07<]\xc9\x14\xe1BA\x06\xbe\x084'
pw = b'BZh91AY&SY\x94$|\x0e\x00\x00\x00\x81\x00\x03$ \x00!\x9ah3M\x13<]\xc9\x14\xe1BBP\x91\xf08'
username = bz2.decompress(un)
password = bz2.decompress(pw)
print(username)
print(password)