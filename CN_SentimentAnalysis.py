import jieba
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_1d,global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical,pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 中文文本分词
def word_cut(mytext):
    return " ".join(jieba.cut(mytext))

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.replace('\n', ' ').split()
    return stopwords_list

def nlp_cnn(trainX,trainY,testX,testY):
    # pad the sequence
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # one_hot encoding
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)
    # # build an embedding
    network = input_data(shape=[None, 100], name='input')
    network = tflearn.embedding(network, input_dim=10000, output_dim=128)
    # build an convnet
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer='L2')
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer='L2')
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer='L2')
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
    # training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True,batch_size=32)

def nlp_rnn(trainX,trainY,testX,testY):
    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)
    # Network building
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch=5, validation_set=(testX, testY), show_metric=True,batch_size=32)

if __name__=='__main__':
    df = pd.read_csv("C:/Users/11245/Desktop/好好看好好学/毕业设计/中文数据集/ChnSentiCorp_htl_all.csv", encoding='utf-8')
    x_data = df['review'].astype(str).to_list()
    y_data = df['label'].astype(int).to_list()
    # # 去除标点符号，进行分词
    # for i in range(len(x_data)):
    #     temp = x_data[i]
    #     temp = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]", "", temp)
    #     x_data[i] = word_cut(temp)
    # # 词向量转换
    # n_features = 100
    # stop_words_file = 'C:/Users/11245/Desktop/好好看好好学/毕业设计/中文数据集/哈工大停用词表.txt'
    # stopwords = get_custom_stopwords(stop_words_file)
    # tf_vectorizer = TfidfVectorizer(strip_accents='unicode', max_features=n_features, stop_words=stopwords,
    #                                 token_pattern=r"(?u)\b\w+\b", min_df=10)
    # k = 3
    # # network build
    # net = tflearn.input_data([None, 100])
    # net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    # net = tflearn.lstm(net, 128, dropout=0.8)
    # net = tflearn.fully_connected(net, 2, activation='softmax')
    # net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    # # Training
    # model = tflearn.DNN(net, tensorboard_verbose=0)
    # # k折交叉验证
    # kf = KFold(n_splits=k, random_state=2020)
    # for train_index, test_index in kf.split(x_data, y_data):
    #     trainX, testX, trainY, testY = [], [], [], []
    #     for i in range(len(train_index)):
    #         trainX.append(x_data[train_index[i]])
    #         trainY.append(y_data[train_index[i]])
    #     for i in range(len(test_index)):
    #         testX.append(x_data[test_index[i]])
    #         testY.append(y_data[test_index[i]])
    #
    #     # 词向量转换
    #     x_train = tf_vectorizer.fit_transform(trainX)
    #     x_test = tf_vectorizer.fit_transform(testX)
    #     trainX = x_train.toarray()
    #     testX = x_test.toarray()
    #     # pad the sequence
    #     trainX = pad_sequences(trainX, maxlen=100, value=0.)
    #     testX = pad_sequences(testX, maxlen=100, value=0.)
    #     # one_hot encoding
    #     trainY = to_categorical(trainY, nb_classes=2)
    #     testY = to_categorical(testY, nb_classes=2)
    #     model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True,
    #               batch_size=32)
    # 去除标点符号，进行分词
    for i in range(len(x_data)):
        temp = x_data[i]
        temp = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]", "", temp)
        x_data[i] = word_cut(temp)
    # 划分训练集/测试集
    x_train, x_test, trainY, testY = train_test_split(x_data, y_data, test_size=0.2, random_state=2020)
    # 词向量转换
    n_features = 1000
    stop_words_file = 'C:/Users/11245/Desktop/好好看好好学/毕业设计/中文数据集/哈工大停用词表.txt'
    stopwords = get_custom_stopwords(stop_words_file)
    tf_vectorizer = TfidfVectorizer(strip_accents='unicode',max_features=n_features,stop_words=stopwords,token_pattern=r"(?u)\b\w+\b",min_df=10)
    x_train = tf_vectorizer.fit_transform(x_train)
    x_test = tf_vectorizer.fit_transform(x_test)
    trainX = x_train.toarray()
    testX = x_test.toarray()
    nlp_rnn(trainX,trainY,testX,testY)
