def charDict(string):
    dict = {}
    for i in string:
        if i not in dict.keys():
            dict[i] = 1
        else:
            dict[i] += 1
    return dict

if __name__ == '__main__':
    a = []
    for i in iter(input, 'end'):
        a.append(i)
    transStr = ''.join(a)
    dataDict = charDict(transStr)
    print(dataDict.items())
