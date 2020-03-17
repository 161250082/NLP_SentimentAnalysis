import pandas as pd
from gensim.models import word2vec
import jieba
import re

def word_cut(mytext):
    return " ".join(jieba.cut(mytext))

# df = pd.read_csv("C:/Users/11245/Desktop/好好看好好学/毕业设计/中文数据集/online_shopping_10_cats.csv", encoding='utf-8')
# x_data = df['review'].astype(str).to_list()
# words = []
# # 去除标点符号，进行分词
# for i in range(len(x_data)):
#     temp = x_data[i]
#     temp = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]", "", temp)
#     x_data[i] = word_cut(temp)
# for i in range(len(x_data)):
#     words.append(x_data[i].split(' '))
# model = word2vec.Word2Vec(words, size=100,sg=1) # 默认window=5
# model.save(u"chineseCorpus.model")
model = word2vec.Word2Vec.load("chineseCorpus.model")
print(model['name'])