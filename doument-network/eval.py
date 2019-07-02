import tensorflow as tf
import numpy as np
import pickle
import time
from model import models
from gensim.models import Word2Vec
import os
def get_data(data):
    data_new = np.zeros([BATCH_SIZE, SEN_SIZE, VECTOR_SIZE], np.float32)
    lenss = []
    masks_de = []
    index = np.random.randint(0, len(data), BATCH_SIZE)
    data = [data[w] for w in index]
    for i in range(len(data)):
        lens = len(data[i])
        for j in range(lens):
            if id2word[data[i][j]]!='UNKNOW':
                data_new[i, j] = wmodel[id2word[data[i][j]]]
        masks_de.append(np.concatenate([np.ones(lens, np.int32), np.zeros(SEN_SIZE - lens, np.int32)]))
        lenss.append(lens)
    blocks = np.random.randint(0,2,(BATCH_SIZE, SEN_SIZE))
    return np.array(data_new), np.array(lenss), np.array(masks_de), blocks

def get_data_test(data):
    data_new = np.zeros([BATCH_SIZE, SEN_SIZE, VECTOR_SIZE], np.float32)
    lenss = []
    masks_de = []
    for i in range(len(data)):
        lens = len(data[i])
        for j in range(lens):
            if id2word[data[i][j]]!='UNKNOW':
                data_new[i, j] = wmodel[id2word[data[i][j]]]
        masks_de.append(np.concatenate([np.ones(lens, np.int32), np.zeros(SEN_SIZE - lens, np.int32)]))
        lenss.append(lens)
    blocks = np.random.randint(0,2,(BATCH_SIZE, SEN_SIZE))
    return np.array(data_new), np.array(lenss), np.array(masks_de), blocks

def build_w2v():
    traindata = np.load('traindata.npy').tolist()
    np.random.shuffle(traindata)
    model = Word2Vec(traindata, sg=1, size=VECTOR_SIZE, window=5, min_count=5, workers=4, iter=20)
    model.save('word2vector.model')
    return model

def read():
    #pre_topics_train_noclip_sigmoidcross_64.npy
    #doc2vec_topics_test_dbow.npy
    binary_codes_test = np.load('pre_topics_test_tanh_square_64.npy')
    binary_codes_train = np.load('pre_topics_train_tanh_square_64.npy')
    train_label = np.load('trainlabel.npy')
    test_label = np.load('testlabel.npy')
    return binary_codes_train, binary_codes_test, train_label, test_label

def countMAP(result, train_label, po, mAP_n):
    AP = 0
    total_relevant = 0
    buffer_yes = np.zeros(mAP_n)
    Ns = np.arange(1, mAP_n+1, 1)
    for i in range(mAP_n):
        if train_label[result[i]] == po:
            buffer_yes[i] = 1
            total_relevant += 1

    P = np.cumsum(buffer_yes)/Ns
    if sum(buffer_yes)!=0:
        AP += sum(P*buffer_yes)/sum(buffer_yes)
    return AP
def countP(result, train_label, po, P_n):
    P = 0
    for i in range(P_n):
        if train_label[result[i]] == po:
            P += 1
    return P/P_n

def eval_total(binary_codes_train, binary_codes_test, train_label, test_label, mAP_n, P_n):
    lens1 = len(binary_codes_test)
    lens2 = len(binary_codes_train)
    P = 0
    mAP = 0
    dist = np.zeros((lens1, lens2), dtype=np.float32)
    binary_codes_train_norm = np.linalg.norm(binary_codes_train, axis=1)
    binary_codes_test_norm = np.linalg.norm(binary_codes_test, axis=1)
    for i in range(lens1):
        dist[i] = -np.sum(binary_codes_test[i] * binary_codes_train, 1)/binary_codes_train_norm/binary_codes_test_norm[i]
    results = []
    for i in range(lens1):
        results.append(np.argsort(dist[i]))
    for i in range(lens1):
        mAP += countMAP(results[i], train_label, test_label[i], mAP_n)
        P += countP(results[i], train_label, test_label[i], P_n)
    return  P/lens1, mAP/lens1

BATCH_SIZE = 32
DIM = 64
HIDDEN_SIZE = 512
SEN_SIZE = 256
VECTOR_SIZE = 100
LAMBDA = 10
ITERS = 15001
traindata = np.load('traindata_vector.npy')
testdata = np.load('testdata_vector.npy')
f = open('diction.pkl','rb')
diction = pickle.load(f)
f.close()
f = open('id2word.pkl','rb')
id2word = pickle.load(f)
f.close()
DIC_SIZE = len(diction)
m = models(HIDDEN_SIZE, HIDDEN_SIZE/2, DIC_SIZE, BATCH_SIZE, SEN_SIZE, DIM, VECTOR_SIZE)
txts = tf.placeholder(tf.float32, [BATCH_SIZE, SEN_SIZE, VECTOR_SIZE])
random_blocks = tf.placeholder(tf.float32, [BATCH_SIZE, SEN_SIZE])
txts_blocked = txts * tf.expand_dims(random_blocks, 2)
sen_size = tf.placeholder(tf.int32, [BATCH_SIZE])
masks_de = tf.placeholder(tf.float32, [BATCH_SIZE, SEN_SIZE])
codes = tf.placeholder(tf.float32, [None, DIM])
labels = tf.placeholder(tf.int32, [None])
y_logits = tf.layers.dense(codes, 20)
y = tf.nn.softmax(y_logits)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf.one_hot(labels, 20), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pre_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=labels))
train_op_ACC = tf.train.AdamOptimizer(1e-3).minimize(pre_loss)
#rebuild_loss
topics = m.Encoder(txts, sen_size)
rebuild_txts, rebuild_loss = m.Decoder(topics, sen_size, txts_blocked, txts, masks_de, random_blocks)
rebuild_loss = tf.reduce_mean(rebuild_loss)

lr = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
decoder_vars = [v for v in vars if 'decoder' in v.name]
encoder_vars = [v for v in vars if 'encoder' in v.name]
train_op_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(rebuild_loss, var_list=decoder_vars + encoder_vars)

def train():
    lr_real = 1e-4
    saver = tf.train.Saver()
    for i in range(ITERS):
        time_start = time.time()
        _txts, _masks_en, _masks_de, blocks = get_data(traindata)
        session.run([train_op_AE],
                    feed_dict={txts: _txts, sen_size: _masks_en, masks_de: _masks_de, lr: lr_real,
                               random_blocks: blocks})
        time_end = time.time()
        if i % 1000 == 0:
            _loss, _topics, _rebuild_txts, _txts_blocked = session.run(
                [rebuild_loss, topics, rebuild_txts, txts_blocked],
                feed_dict={txts: _txts, sen_size: _masks_en,
                           masks_de: _masks_de, lr: lr_real,
                           random_blocks: blocks})
            print('loss:%f' % (_loss))
            print('totally cost', time_end - time_start)
        if i % 1000 == 0:
            saver.save(session, './gan_tanh_square_%d' % DIM)
            print('model saved')

def test():
    sp = './gan_tanh_square_%d' % DIM
    pre_topics = []
    for i in range(int(len(traindata) / BATCH_SIZE) + 1):
        data = traindata[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, len(traindata))].tolist()
        tlen = len(data)
        for k in range(BATCH_SIZE - tlen):
            data.append(data[0])
        _txts, _masks_en, _masks_de, blokss = get_data_test(data)
        _topics = session.run(topics, feed_dict={txts: _txts, sen_size: _masks_en, masks_de: _masks_de})
        pre_topics.append(_topics[0:min(BATCH_SIZE, len(traindata) - i * BATCH_SIZE)])
    pre_topics = np.concatenate(pre_topics, 0)
    np.save('pre_topics_train%s.npy' % sp[5:], pre_topics)

    pre_topics = []
    for i in range(int(len(testdata) / BATCH_SIZE) + 1):
        data = testdata[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, len(testdata))].tolist()
        tlen = len(data)
        for k in range(BATCH_SIZE - tlen):
            data.append(data[0])
        _txts, _masks_en, _masks_de, blokss = get_data_test(data)
        _topics = session.run(topics, feed_dict={txts: _txts, sen_size: _masks_en, masks_de: _masks_de})
        pre_topics.append(_topics[0:min(BATCH_SIZE, len(testdata) - i * BATCH_SIZE)])
    pre_topics = np.concatenate(pre_topics, 0)
    np.save('pre_topics_test%s.npy' % sp[5:], pre_topics)

def eval_acc(codes_train, codes_test, train_label, test_label):
    for i in range(5001):
        _, _pre_loss, _accuracy = session.run([train_op_ACC, pre_loss, accuracy],
                                              feed_dict={codes: codes_train, labels: train_label})
        if i % 100 == 0:
            print(i)
            _, _accuracy_test = session.run([pre_loss, accuracy], feed_dict={codes: codes_test, labels: test_label})
            print('loss:%f, acc_train:%f, acc_test:%f' % (_pre_loss, _accuracy, _accuracy_test))

    return _accuracy_test

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    for i in range(1):
        wmodel = build_w2v()
        session.run(tf.initialize_all_variables())
        train()
        test()
        codes_train, codes_test, train_label, test_label = read()
        rs = eval_acc(codes_train, codes_test, train_label, test_label)
        rss_ac.append(rs)
        print('Acc:')
        print(rss_ac)
        test_list = [25, 50, 100]
        rs_retrieval = []
        for s in test_list:
            rs = eval_total(codes_train, codes_test, train_label, test_label, s, s)
            rs_retrieval.append(rs)
        rss_re.append(rs_retrieval)
        print('P, mAP:')
        print(rss_re)
