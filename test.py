import pandas as pd
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import jieba.analyse

def textRank4zh():
    # 加载数据
    df = pd.read_csv("C:/Users/11245/Desktop/好好看好好学/毕业设计/中文数据集/ChnSentiCorp_htl_all.csv", encoding='utf-8')
    x_data = df['review'].astype(str).to_list()
    # 创建分词类的实例
    tr4w = TextRank4Keyword()
    # 对文本进行分析，设定窗口大小为2，并将英文单词小写
    for i in range(0, 100):
        print(x_data[i])
        tr4w.analyze(text=x_data[i], lower=True, window=2)
        # 从关键词列表中获取前20个关键词
        print('关键词为：', end='')
        for item in tr4w.get_keywords(num=5, word_min_len=1):
            # 打印每个关键词的内容及关键词的权重
            print(item.word, item.weight, ' ', end='')
        print('\n')
        print('关键短语为：', end='')
        # 从关键短语列表中获取关键短语
        for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num=1):
            print(phrase, ' ', end='')
        print('\n')
if __name__=='__main__':
    # 加载数据
    df = pd.read_csv("C:/Users/11245/Desktop/好好看好好学/毕业设计/中文数据集/ChnSentiCorp_htl_all.csv", encoding='utf-8')
    x_data = df['review'].astype(str).to_list()
    print(x_data[0])
    # textrank
    keywords_textrank = jieba.analyse.textrank(x_data[0])
    print(keywords_textrank)
    # tf-idf
    keywords_tfidf = jieba.analyse.extract_tags(x_data[0])
    print(keywords_tfidf)