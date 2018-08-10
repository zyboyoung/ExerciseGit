def func(str):
    str = chr(ord(str)+2)
    if str > 'z':
        str = chr(ord(str) - ord('z') - 1 + ord('a'))
    else:
        pass
    return str

if __name__=='__main__':
    secret = input()
    valueList = list(map(func, list(secret)))
    result = ''
    for i in range(0, len(valueList)):
        result += str(valueList[i])
    print(result)
