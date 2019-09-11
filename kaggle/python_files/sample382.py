import re
from time import time
from collections import Counter

import tensorflow as tf
import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from fastcache import clru_cache as lru_cache


t_start = time()

stemmer = PorterStemmer()

@lru_cache(1024)
def stem(s):
    return stemmer.stem(s)

whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')

def tokenize(text):
    text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        t = stem(t)
        tokens.append(t)

    return tokens

class Tokenizer:
    def __init__(self, min_df=10, tokenizer=str.split):
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None

    def fit_transform(self, texts):
        tokenized = []
        doc_freq = Counter()
        n = len(texts)

        for text in texts:
            sentence = self.tokenizer(text)
            tokenized.append(sentence)
            doc_freq.update(set(sentence))

        vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
        doc_freq = [doc_freq[t] for t in vocab]

        self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        max_len = 0
        result_list = []
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len = max(max_len, len(text))
            result_list.append(text)

        self.max_len = max_len
        result = np.zeros(shape=(n, max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result    

    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text)[:self.max_len]
            result[i, :len(text)] = text

        return result
    
    def vocabulary_size(self):
        return len(self.vocab) + 1


print('reading train data...')
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_train = df_train[df_train.price != 0].reset_index(drop=True)

price = df_train.pop('price')
y = np.log1p(price.values)
mean = y.mean()
std = y.std()
y = (y - mean) / std
y = y.reshape(-1, 1)

df_train.name.fillna('unkname', inplace=True)
df_train.category_name.fillna('unk_cat', inplace=True)
df_train.brand_name.fillna('unk_brand', inplace=True)
df_train.item_description.fillna('nodesc', inplace=True)

print('processing category...')

def paths(tokens):
    all_paths = ['/'.join(tokens[0:(i+1)]) for i in range(len(tokens))]
    return ' '.join(all_paths)

@lru_cache(1024)
def cat_process(cat):
    cat = cat.lower()
    cat = whitespace.sub('', cat)
    split = cat.split('/')
    return paths(split)

df_train.category_name = df_train.category_name.apply(cat_process)

cat_tok = Tokenizer(min_df=50)
X_cat = cat_tok.fit_transform(df_train.category_name)
cat_voc_size = cat_tok.vocabulary_size()


print('processing title...')

name_tok = Tokenizer(min_df=10, tokenizer=tokenize)
X_name = name_tok.fit_transform(df_train.name)
name_voc_size = name_tok.vocabulary_size()


print('processing description...')

desc_num_col = 40
desc_tok = Tokenizer(min_df=50, tokenizer=tokenize)
X_desc = desc_tok.fit_transform(df_train.item_description)
X_desc = X_desc[:, :desc_num_col]
desc_voc_size = desc_tok.vocabulary_size()


print('processing brand...')

df_train.brand_name = df_train.brand_name.str.lower()
df_train.brand_name = df_train.brand_name.str.replace(' ', '_')

brand_cnt = Counter(df_train.brand_name[df_train.brand_name != 'unk_brand'])
brands = sorted(b for (b, c) in brand_cnt.items() if c >= 50)
brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

X_brand = df_train.brand_name.apply(lambda b: brands_idx.get(b, 0))
X_brand = X_brand.values.reshape(-1, 1) 
brand_voc_size = len(brands) + 1


print('processing other features...')

X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)


print('defining the model...')

def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i+step])
    return res

def conv1d(inputs, num_filters, filter_size, padding='same'):
    he_std = np.sqrt(2 / (filter_size * num_filters))
    out = tf.layers.conv1d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size,
        activation=tf.nn.relu, 
        kernel_initializer=tf.random_normal_initializer(stddev=he_std))
    return out

def dense(X, size, reg=0.0, activation=None):
    he_std = np.sqrt(2 / int(X.shape[1]))
    out = tf.layers.dense(X, units=size, activation=activation, 
                     kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                     kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
    return out

def embed(inputs, size, dim):
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup


name_embeddings_dim = 32
name_seq_len = X_name.shape[1]
desc_embeddings_dim = 32
desc_seq_len = X_desc.shape[1]

brand_embeddings_dim = 4

cat_embeddings_dim = 12
cat_seq_len = X_cat.shape[1]


graph = tf.Graph()
graph.seed = 1

with graph.as_default():
    place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
    place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
    place_brand = tf.placeholder(tf.int32, shape=(None, 1))
    place_cat = tf.placeholder(tf.int32, shape=(None, cat_seq_len))
    place_ship = tf.placeholder(tf.float32, shape=(None, 1))
    place_cond = tf.placeholder(tf.uint8, shape=(None, 1))

    place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    place_lr = tf.placeholder(tf.float32, shape=(), )

    name = embed(place_name, name_voc_size, name_embeddings_dim)
    desc = embed(place_desc, desc_voc_size, desc_embeddings_dim)
    brand = embed(place_brand, brand_voc_size, brand_embeddings_dim)
    cat = embed(place_cat, cat_voc_size, cat_embeddings_dim)

    name = conv1d(name, num_filters=10, filter_size=3)
    name = tf.layers.dropout(name, rate=0.5)
    name = tf.contrib.layers.flatten(name)
    print(name.shape)

    desc = conv1d(desc, num_filters=10, filter_size=3)
    desc = tf.layers.dropout(desc, rate=0.5)
    desc = tf.contrib.layers.flatten(desc)
    print(desc.shape)

    brand = tf.contrib.layers.flatten(brand)
    print(brand.shape)

    cat = tf.layers.average_pooling1d(cat, pool_size=cat_seq_len, strides=1, padding='valid')
    cat = tf.contrib.layers.flatten(cat)
    print(cat.shape)
    
    ship = place_ship
    print(ship.shape)

    cond = tf.one_hot(place_cond, 5)
    cond = tf.contrib.layers.flatten(cond)
    print(cond.shape)

    out = tf.concat([name, desc, brand, cat, ship, cond], axis=1)
    print('concatenated dim:', out.shape)

    out = dense(out, 100, activation=tf.nn.relu)
    out = tf.layers.dropout(out, rate=0.5)
    out = dense(out, 1)

    loss = tf.losses.mean_squared_error(place_y, out)
    rmse = tf.sqrt(loss)
    opt = tf.train.AdamOptimizer(learning_rate=place_lr)
    train_step = opt.minimize(loss)

    init = tf.global_variables_initializer()

session = tf.Session(config=None, graph=graph)
session.run(init)


print('training the model...')

for i in range(4):
    t0 = time()
    np.random.seed(i)
    train_idx_shuffle = np.arange(X_name.shape[0])
    np.random.shuffle(train_idx_shuffle)
    batches = prepare_batches(train_idx_shuffle, 500)

    if i <= 2:
        lr = 0.001
    else:
        lr = 0.0001

    for idx in batches:
        feed_dict = {
            place_name: X_name[idx],
            place_desc: X_desc[idx],
            place_brand: X_brand[idx],
            place_cat: X_cat[idx],
            place_cond: X_item_cond[idx],
            place_ship: X_shipping[idx],
            place_y: y[idx],
            place_lr: lr,
        }
        session.run(train_step, feed_dict=feed_dict)

    took = time() - t0
    print('epoch %d took %.3fs' % (i, took))


print('reading the test data...')

df_test = pd.read_csv('../input/test.tsv', sep='\t')

df_test.name.fillna('unkname', inplace=True)
df_test.category_name.fillna('unk_cat', inplace=True)
df_test.brand_name.fillna('unk_brand', inplace=True)
df_test.item_description.fillna('nodesc', inplace=True)

df_test.category_name = df_test.category_name.apply(cat_process)
df_test.brand_name = df_test.brand_name.str.lower()
df_test.brand_name = df_test.brand_name.str.replace(' ', '_')

X_cat_test = cat_tok.transform(df_test.category_name)
X_name_test = name_tok.transform(df_test.name)

X_desc_test = desc_tok.transform(df_test.item_description)
X_desc_test = X_desc_test[:, :desc_num_col]

X_item_cond_test = (df_test.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
X_shipping_test = df_test.shipping.astype('float32').values.reshape(-1, 1)

X_brand_test = df_test.brand_name.apply(lambda b: brands_idx.get(b, 0))
X_brand_test = X_brand_test.values.reshape(-1, 1)


print('applying the model to test...')

n_test = len(df_test)
y_pred = np.zeros(n_test)

test_idx = np.arange(n_test)
batches = prepare_batches(test_idx, 5000)

for idx in batches:
    feed_dict = {
        place_name: X_name_test[idx],
        place_desc: X_desc_test[idx],
        place_brand: X_brand_test[idx],
        place_cat: X_cat_test[idx],
        place_cond: X_item_cond_test[idx],
        place_ship: X_shipping_test[idx],
    }
    batch_pred = session.run(out, feed_dict=feed_dict)
    y_pred[idx] = batch_pred[:, 0]

y_pred = y_pred * std + mean
y_pred = np.expm1(y_pred)


print('writing the results...')

df_out = pd.DataFrame()
df_out['test_id'] = df_test.test_id
df_out['price'] = y_pred

df_out.to_csv('submission_tf.csv', index=False)


t_end = time()
took = (t_end - t_start) / 60
print('done in %.3f minutes' % took)

