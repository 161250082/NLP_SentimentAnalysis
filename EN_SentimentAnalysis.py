import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_1d,global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical,pad_sequences
import keras
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# 数据预处理，清理特殊字符和停用词（文本中一些高频的代词连词介词）（停用词表）
# 中文文本分词（词粒度或字粒度）
# 词向量（word2vec）
# CNN/RNN等深度学习网络及其变体解决自动特征提取

# IMDB data loading
# train , test = imdb.load_data(path='imdb.pkl' , n_words=10000 , valid_portion=0.1)
# trainX , trainY = train
# testX , testY = test

def nlp_cnn(trainX,trainY,testX,testY):
    # pad the sequence
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # one_hot encoding
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)
    # build an embedding
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
    model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)

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
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,batch_size=32)

if __name__=='__main__':
    (trainX, trainY), (testX, testY)=keras.datasets.imdb.load_data(path="C:/Users/11245/Desktop/好好看好好学/毕业设计/NLP数据集/imdb.npz", num_words=10000)
    # 划分数据集，进行k折交叉验证(常用值3、6、10)
    k = 6
    x_data = []
    x_data.extend(trainX)
    x_data.extend(testX)
    y_data = []
    y_data.extend(trainY)
    y_data.extend(testY)
    # # # build an embedding
    # network = input_data(shape=[None, 100], name='input')
    # network = tflearn.embedding(network, input_dim=10000, output_dim=128)
    # # build an convnet
    # branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer='L2')
    # branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer='L2')
    # branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer='L2')
    # network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    # network = tf.expand_dims(network, 2)
    # network = global_max_pool(network)
    # network = dropout(network, 0.5)
    # network = fully_connected(network, 2, activation='softmax')
    # network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
    # # training
    # model = tflearn.DNN(network, tensorboard_verbose=0)

    # Network building
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    # k折交叉验证
    kf = KFold(n_splits=k, random_state=2020)
    for train_index,test_index in kf.split(x_data, y_data):
        trainX,testX,trainY,testY = [],[],[],[]
        for i in range(len(train_index)):
            trainX.append(x_data[train_index[i]])
            trainY.append(y_data[train_index[i]])
        for i in range(len(test_index)):
            testX.append(x_data[test_index[i]])
            testY.append(y_data[test_index[i]])
        # nlp_rnn(trainX,trainY,testX,testY)
        # pad the sequence
        trainX = pad_sequences(trainX, maxlen=100, value=0.)
        testX = pad_sequences(testX, maxlen=100, value=0.)
        # one_hot encoding
        trainY = to_categorical(trainY, nb_classes=2)
        testY = to_categorical(testY, nb_classes=2)
        model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True,batch_size=32)