import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
class models:
    def __init__(self, h,h2,v,b,l, d, vd):
        self.h = h
        self.h2 = h2
        self.v = v
        self.b = b
        self.l = l
        self.d = d
        self.vd = vd
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    def corss_loss(self, a, b):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=tf.nn.sigmoid(b)),1)
    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.h)

    def Encoder(self, txts, sen_size, reuse = False):
                with tf.variable_scope('encoder', reuse=reuse):
            lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h, reuse=False)
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h, reuse=False)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2])
            state = lstm_cell.zero_state(self.b, tf.float32)
            with tf.variable_scope('lstm'):
                _, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=txts, dtype=tf.float32, sequence_length=sen_size, initial_state=state)
                final_out = tf.tanh(tf.layers.dense(tf.concat([final_state[0][0], final_state[1][0],final_state[0][1], final_state[1][1]], 1), self.d, name='outputs'))
            return final_out


    def Decoder(self, topics, sen_size, txts_blocked, txts, masks, random_blocks, reuse = False):
        with tf.variable_scope('decoder', reuse=reuse):
            ans_id = []
            loss = 0
            lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h2, reuse=False)
            state1 = self.get_initial_lstm(topics, 'state1')
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h2, reuse=False)
            state2 = self.get_initial_lstm(topics, 'state2')
            with tf.variable_scope('lstm'):
                topicss = tf.concat([tf.expand_dims(topics, 1) for i in range(self.l)], 1)
                inputs = tf.concat([topicss, txts_blocked], 2)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell1, cell_bw=lstm_cell2, inputs=inputs, initial_state_fw=state1, initial_state_bw=state2, sequence_length=sen_size)
                outputs = tf.concat([outputs[0], outputs[1]], axis=2)
                for i in range(self.l):
                    ans_logits = tf.layers.dense(outputs[:,i], self.vd, name='v2w', reuse=(i != 0))
                    ans_tanh = tf.tanh(ans_logits)
                    txt_tanh = tf.tanh(txts[:, i])
                    loss += (1 - random_blocks[:, i]) * masks[:, i] * tf.reduce_mean(tf.square(ans_tanh - txt_tanh), 1)
                    ans_id.append(ans_logits)

            return  ans_id, loss


    def get_initial_lstm(self, topics, name):
        with tf.variable_scope('initial_lstm_'+name):
            h = tf.nn.tanh(tf.layers.dense(topics, self.h2))
            c = tf.nn.tanh(tf.layers.dense(topics, self.h2))
            return  LSTMStateTuple(c,h)
