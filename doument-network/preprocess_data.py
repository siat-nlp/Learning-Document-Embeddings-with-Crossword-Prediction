import numpy as np
import re
import pickle
def first():
    f = open('20ng-train-stemmed.txt', encoding='UTF-8')
    txts = f.read()
    f.close()
    txts = txts.split('\n')
    txts2 = []
    for tx in txts:
        txts2.append(re.split('[ \t]', tx))
    classes = {}
    count = 0
    txts2 = txts2[0:len(txts2) - 1]
    for tx in txts2:
        label = tx[0]
        if label not in classes.keys():
            classes[label] = count
            count += 1
    data = []
    label = []
    for tx in txts2:
        data.append(tx[1:])
        label.append(classes[tx[0]])
    data = np.array(data)
    label = np.array(label)
    f = open('classes.pkl', 'wb+')
    pickle.dump(classes, f)
    np.save('traindata.npy', data)
    np.save('trainlabel.npy', label)

def second():
    f = open('20ng-test-stemmed.txt', encoding='UTF-8')
    txts = f.read()
    f.close()
    f = open('diction.pkl', 'rb')
    diction = pickle.load(f)
    f.close()
    f = open('id2word.pkl', 'rb')
    id2word = pickle.load(f)
    f.close()
    f = open('classes.pkl', 'rb')
    classes = pickle.load(f)
    f.close()
    txts = txts.split('\n')
    txts2 = []
    for tx in txts:
        txts2.append(re.split('[ \t]', tx))
    txts2 = txts2[0:len(txts2) - 1]
    data = []
    label = []
    for tx in txts2:
        data.append(tx[1:])
        label.append(classes[tx[0]])
    data = np.array(data)
    label = np.array(label)
    np.save('testdata.npy', data)
    np.save('testlabel.npy', label)

def third():
    MAX_LENGTH = 256
    traindata = np.load('traindata.npy')
    diction = {}
    new_traindata = []
    maxs = 1
    count = 1
    for sen in traindata:
        for word in sen:
            word = re.sub('[^a-zA-Z]', '', word)
            if word in diction.keys():
                diction[word] += 1
            else:
                diction[word] = 1
    old_list = list(diction.keys())
    for key in old_list:
        if diction[key] < 5:
            del diction[key]
    diction_list = sorted(diction.items(), key=lambda d: d[1], reverse=True)
    for key in diction_list:
        diction[key[0]] = count
        count += 1
    diction['END'] = count
    diction['UNKNOW'] = 0

    for sen in traindata:
        new_sen = []
        for word in sen:
            word = re.sub('[^a-zA-Z]', '', word)
            if word in diction.keys():
                new_sen.append(diction[word])
            else:
                new_sen.append(diction['UNKNOW'])
        new_sen = np.array(new_sen).astype(np.int32)
        lens = len(new_sen)
        if (lens >= MAX_LENGTH):
            new_sen = new_sen[0:MAX_LENGTH]

        new_traindata.append(new_sen)
    new_traindata = np.array(new_traindata)
    np.save('traindata_vector.npy', new_traindata)
    id2word = {}
    for key in diction.keys():
        id2word[diction[key]] = key
    f = open('diction.pkl', 'wb+')
    pickle.dump(diction, f)
    f = open('id2word.pkl', 'wb+')
    pickle.dump(id2word, f)

def forth():
    MAX_LENGTH = 256
    testdata = np.load('testdata.npy')
    new_testdata = []
    f = open('diction.pkl', 'rb')
    diction = pickle.load(f)
    f.close()
    f = open('id2word.pkl', 'rb')
    id2word = pickle.load(f)
    f.close()
    for sen in testdata:
        new_sen = []
        for word in sen:
            word = re.sub('[^a-zA-Z]', '', word)
            if word in diction.keys():
                new_sen.append(diction[word])
            else:
                new_sen.append(diction['UNKNOW'])
        new_sen = np.array(new_sen).astype(np.int32)
        lens = len(new_sen)
        if (lens >= MAX_LENGTH):
            new_sen = new_sen[0:MAX_LENGTH]

        new_testdata.append(new_sen)
    new_testdata = np.array(new_testdata)
    np.save('testdata_vector.npy', new_testdata)
first()
second()
third()
forth()
