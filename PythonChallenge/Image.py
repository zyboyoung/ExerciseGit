import re
from PIL import Image

img = Image.open(r'D:\Python\Exercise\oxygen.PNG')
data = img.convert('L').getdata()  

message = []  
for i in range(3,608,7):  
        message.append(chr(data[img.size[0]*50+i]))

result = ''.join(message)

pattern = re.compile(r'\d+\d')
result_data = pattern.findall(result)

final_data = []
for i in range(len(result_data)):
    final_data.append(chr(int(result_data[i])))
print(''.join(final_data))