import zipfile

root_dir = 'D:\\Python\\exercise\\zip\\'
end_dir = '.txt'
start = '90052'


def extract(data_num):
    with open(root_dir + str(data_num) + end_dir) as f:
        f = f.readline()
        # print(f)
        data = f.split()
        return data[-1]

name_list = []

while(str.isdigit(start)):
    start = extract(start)
    name_list.append(start)
# print(name_list)

zip_file = zipfile.ZipFile(root_dir + 'channel.zip')
comments = {}
for name in zip_file.namelist():
    comments[name] = zip_file.getinfo(name).comment
# print(comments)

# zz = ''

zz = []
for i in name_list:
    if 'comments' not in i:
    #     zz += str(comments[i + '.txt'].decode('utf-8'))
    	zz.append(str(comments[i + '.txt'].decode('utf-8')))

print(''.join(zz))
