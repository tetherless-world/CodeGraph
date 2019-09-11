import collections
import os


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000

class myConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 3
    num_steps = 35
    hidden_size = 1500
    max_epoch = 32
    max_max_epoch = 128
    keep_prob = 0.25
    lr_decay = 1 / 1.15
    batch_size = 64
    vocab_size = 10000

# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------


import time
import numpy as np
import tensorflow as tf


class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps

        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = ptb_producer(data, batch_size,
                                                     num_steps, name=name)

class PTBModel(object):

    def __init__(self, is_training, _config, _input):
        self._input = _input

        self.batch_size = _input.batch_size
        self.num_steps = _input.num_steps
        self.size = _config.hidden_size
        self.vocab_size = config.vocab_size
        self.get_rnn(is_training, _config)

    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=0.0,
                                       state_is_tuple=True,
                                       name='basic_lstm_cell',
                                      activation=tf.nn.leaky_relu)
    def attn_cell(self, keep_prob):
        return tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell(), output_keep_prob=keep_prob)

    def Dense(self, x, filter_x, filter_y, name):
        w = tf.get_variable(name+'w',
                            [filter_x, filter_y],
                            dtype=tf.float32)
        b = tf.get_variable(name+'b',
                            [filter_y],
                            dtype=tf.float32)

        x = tf.matmul(x, w) + b
        
        if name.startswith('relu'):
            x = tf.nn.relu(x)
            
        return x

    def get_rnn(self, is_training, _config):

        if is_training and _config.keep_prob < 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([self.attn_cell(_config.keep_prob) for _ in range(_config.num_layers)],
                                               state_is_tuple=True)
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(_config.num_layers)],
                                               state_is_tuple=True)
                                               
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [self.vocab_size, self.size],
                                    dtype=tf.float32)

        x = tf.nn.embedding_lookup(embedding, self._input.input_data)

        if is_training and _config.keep_prob < 1:
            x = tf.nn.dropout(x, _config.keep_prob)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                (cell_output, state) = cell(x[:, time_step, :], state)
                outputs.append(cell_output)
                
        # flatten
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.size])
        # NN Layer
        y = self.Dense(output, self.size, self.vocab_size, 'relu1')
        y = tf.nn.dropout(y, 0.5)
        y = self.Dense(y, self.vocab_size, self.vocab_size, 'softmax1')
        # y = self.Dense(y, self.vocab_size, self.vocab_size, 'softmax1')
        # y = self.Dense(output, self.size, self.vocab_size, 'relu1')
        # y = self.Dense(y, self.vocab_size, self.vocab_size, 'softmax1')

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([y],
                                                                  [tf.reshape(self._input.targets, [-1])],
                                                                  [tf.ones([self.batch_size * self.num_steps],
                                                                           dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / self.batch_size
        self.final_state = state

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)

        
        
        # Alr = 1.0
        # grad_norm = 5
        # _config.max_grad_norm = 5
        
        # optimizer = tf.train.RMSPropOptimizer(self.lr)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), _config.max_grad_norm)
        capped_gvs = zip(grads, tvars)
                                          
        self.train_op = optimizer.apply_gradients(capped_gvs,
                                                  global_step=tf.train.get_or_create_global_step())
        # self.train_op = optimizer.minimize(self.cost, var_list=tf.trainable_variables())

        self.new_lr = tf.placeholder(tf.float32, shape=[], 
                               name='new_learning_rate')
        self.lr_update = tf.assign(self.lr, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    cost_list = []
    per_list = []

    for step in range(model._input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        costs += vals["cost"]
        iters += model._input.num_steps

        cost_list.append(vals["cost"])
        per_list.append(np.exp(costs / iters))
        
        if verbose and step % (model._input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model._input.epoch_size, np.exp(costs / iters),
                   iters * model._input.batch_size / (time.time() - start_time)))
        
    return np.exp(costs / iters), np.sum(cost_list)/model._input.epoch_size, np.sum(per_list)/model._input.epoch_size

from matplotlib import pyplot as plt

def draw_result(history, test):

    history['cost']
    history['cost']
    epochs = range(len(history['cost']))
    
    plt.plot(epochs, history['cost'], '-o', label='Training cost')
    plt.plot(epochs, history['val_cost'], '-o', label='Validation cost')
    plt.title('Training and validation cost')
    plt.legend()
    plt.savefig('cost.png')

    plt.figure()
    plt.plot(epochs, history['per'], '-o', label='Training Perplexity')
    plt.plot(epochs, history['val_per'], '-o', label='Validation Perplexity')
    plt.title('Training and validation Perplexity')
    plt.legend()
    plt.savefig('per.png')
    
    for i in history:
        history[i] = history[i][3:]
        
    epochs = range(len(history['cost']))
    
    plt.figure()
    plt.plot(epochs, history['cost'], '-o', label='Training cost')
    plt.plot(epochs, history['val_cost'], '-o', label='Validation cost')
    plt.title('Training and validation cost2')
    plt.legend()
    plt.savefig('cost2.png')

    plt.figure()
    plt.plot(epochs, history['per'], '-o', label='Training Perplexity')
    plt.plot(epochs, history['val_per'], '-o', label='Validation Perplexity')
    plt.title('Training and validation Perplexity2')
    plt.legend()
    plt.savefig('per2.png')
    
    plt.figure()
    plt.title(test)
    plt.legend()
    plt.savefig('test.png')
    
raw_data = ptb_raw_data('../input/simple-examples/simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = myConfig()
eval_config = myConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data,
                               name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, _config=config, _input=train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data,
                               name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, _config=config,
                              _input=valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data,
                              name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, _config=eval_config,
                             _input=test_input)

    cost = []
    val_cost = []
    per = []
    val_per = []
    
    logdir = 'tmp'
    sv = tf.train.Supervisor(logdir=logdir)
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i+1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
            train_perplexity, _cost, _per = run_epoch(session, m, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i+1, train_perplexity))
            valid_perplexity, _val_cost, _val_per = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i+1, valid_perplexity))
            
            cost.append(_cost)
            val_cost.append(_val_cost)
            per.append(_per)
            val_per.append(_val_per)

        test_perplexity, _, _ = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)

        history = {'cost':cost, 'val_cost':val_cost, 'per':per, 'val_per':val_per}
        draw_result(history, "Test Perplexity: %.3f" % test_perplexity)
        sv.saver.save(session, 'homework.model', global_step=sv.global_step)
    
