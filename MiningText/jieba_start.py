import jieba

txt = '大夏天的，开着空调，吃着西瓜，刷着微信，敲着代码，别提有多酸爽！'

# jieba.cut(text)进行分词处理，返回的是一个生成器
txt_cut = jieba.cut(txt)
result = ('/'.join(txt_cut))
print(result)

# jieba.add_word()人工添加词汇作为词库，提高拆分准确率
jieba.add_word(word='吃着')
jieba.add_word(word='刷着')
jieba.add_word(word='敲着')
jieba.add_word(word='酸爽')

# jieba.cut(text)进行分词处理，返回的是一个生成器
txt_cut = jieba.cut(txt)
result = ('/'.join(txt_cut))
print(result)

txt_cut = jieba.cut(txt)
# 删除一些不需要的词
result2 = [w for w in txt_cut if w not in ['的', '有', '多', '大', '，', '！']]
print(result2)
