#!/usr/bin/env python
# coding: utf-8

# #### Quick, Draw! Doodle Recognition Challenge
# * dataset은 countrycode, drawing, key_id, recognized, timestamp, word로 구성된다.
#   * countrycode : 접속한 나라
#   * drawing : 유저가 시간에 따라 그린 그림. 예를 들어 10번에 걸쳐서 부엉이를 그린다고 했을때 첫번째로 동그란 원을 그리고 
#   두번째로 부엉이 귀를 그리고 세번째 부엉이 눈을 그리고 ... 이와 같이 10번에 걸쳐서 그릴때 drawing의 데이터는 10개의 row가 
#   생긴다. 한 row에는 한번 그릴때 그린 그림(혹은 stroke)에 대한 x,y 좌표가 담긴다.
#   * key_id : 그림에 대한 id
#   * recognized : AI가 인식했는지 여부
#   * timestamp : 시작한 시간
#   * word : 정답
#   * 자세한 정보는 [이 글](https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset)을 참고한다.
# * dataset은 raw와 simplified, 두 종류가 있다.
#   * raw : 유저가 그린 stroke를 정확히 기록한다.
#   * simplified : raw dataset을 간단하게 요약했다. 예를 들어 raw의 경우 한 직선을 그리고 이것을 8개의 점을 기록한다. 하지만 simplified는 한 직선을 표현하는데 2개의 점만 필요하므로 2개를 기록한다. 같은 정보에 대해 효율적이다.
# * 여러 개의 dataset으로 이루어져 있다. 총 340개의 csv로 구성되며 각 class에 대한 data이다.
# * 340개의 csv의 data size는 약 5천만이다.
#   * AI가 인식한 data의 갯수 : 45M
# * 모든 data를 memory에 올릴 수 없으므로 각 class 마다 일정한 개수를 가져왔다.
#   * 예를 들어 한 class 마다 2000개의 data를 가져오면 총 dataset의 크기는 340 * 2000이 된다.
# * dataset을 train : validation : test = 0.64 : 0.16 : 0.2의 비율로 나누었다.
#   * validation은 model의 train에 대한 overfitting을 확인하기 위해 분리했다.
# * 다음은 Quick, Draw! Doodle Recognition Challenge를 하면서 고민한 사항을 나열했다.
#   * reproducible ***
#     * [이 글](https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development)을 참고했지만 실패했다.
#   * data pipeline은 어떻게 구현할 것인가? 
#     * multiprocessing 사용 불가
#       * memory를 release 못함(??)
#       * allocate memory error
#     * stroke에서 ndarray로 변환하는 과정이 너무 느리다.
#       * 어디서 bottleneck이 발생한 것인가?
#       * O(NK^2), N: data size, K: stroke, points
#     * 병렬처리 가능하고 대용량 data handling할 수 있는 dask package 사용했다.
#   * 약 50M의 data를 어떻게 학습할 것인가?
#     * 모든 data를 사용할 것인가?
#     * 일단 small data부터 train accuracy 높여서 data 양을 늘린다.
#   * recognized = True인 data만 사용할 것인가?
#   * image의 크기는 어떻게 정할 것인가?
#   * model의 구조는 어떻게 할 것인가?
#     * 처음에는 간단한 model부터 시작
#     * 기성 model 사용?
#       * inception, resnet... [etc](https://github.com/keras-team/keras-applications)
#   * hyper-parameter search
#     * learning rate, drop out, regularization rate
#     * heuristic selection
#   * raw, simplified data 중 어느 것을 사용할 것인가?
#     * simplified data 선택
#     * 충분히 image를 표현할 수 있다고 판단
#   * early-stopping
# * model architecture 과정
#   * layer를 적게 사용하여 train set를 완전히 학습한다.
#   * overfitting을 줄인다.
#   * data를 늘린다.
# * customized model
#   * (conv 3x3+1 32, maxpool 2x2+2), dense340
#     * data : 340 * 10
#     * 64x64
#     * #params : 11M
#     * train top3 : 1.0, val top3 : 0.14, test top3 : 0.15
#     * train loss와 val loss를 graph로 확인한 결과 학습을 계속할수록 train loss는 떨어지고 val loss는 감소하다가 증가했다.
#     * overfitting 발생
#   * BN, (dropout 0.2, conv 3x3+1 8, maxpool 2x2+2), (conv 3x3+1 16, maxpool 2x2+2), dense680, dense 340
#     * data : 340 * 10
#     * 64x64
#     * #params : 3M
#     * train top3 1.0, val top_3: 0.1869, test top3 : 0.1889
#     * raw image에 BatchNormalization과 dropout 적용 후 top3가 약간 증가했다.
#       * image를 구분하는 feature는 모든 pixel을 보고 결정하는게 아니라 일부의 중요한 특징(latent feature)을 탐지하여 분류하기 때문에 일부 pixel을 제거해도(dropout 적용하여 일부 pixel을 0으로 만듬) 분류할 수 있다.
#     * overfitting 발생
#   * BN, (dropout 0.2, conv 3x3+1 16, maxpool 2x2+2), (conv 3x3+1 32, maxpool 2x2+2), (conv 3x3+1 64, maxpool 2x2+2), (conv 3x3+1 128, maxpool 2x2+2), (conv 3x3+1 256, maxpool 2x2+2), dense 340
#     * data : 340 * 10
#     * 64x64
#     * #params : 740K
#     * train top3 : 1.0, val top3: 0.1727, test top3 : 0.2372
#       * l2 regularization rate 1e-3
#         * convolution 1~5
#         * l1은 일부 parameter만 남기고 나머지를 0으로 근사한다. 일부 parameter에 의존하므로 unseen data에 대해 robust하지 않다.
#         * l2는 모든 parameter를 골고루 penalty를 주어 l1보다 robust하다. 모든 parameter의 크기가 비슷하다.
#     * overfitting 발생
#   * BN, (dropout 0.2, conv 3x3+1 16, BN, maxpool 2x2+2), (dropout 0.2, conv 3x3+1 32, BN, maxpool 2x2+2), (dropout 0.2 conv 3x3+1 64, BN, maxpool 2x2+2), (dropout 0.2, conv 3x3+1 128, BN, maxpool 2x2+2), (dropout 0.2, conv 3x3+1 256, BN, maxpool 2x2+2), dense 340
#     * data : 340 * 100
#     * 64x64
#     * #params : 741K
#     * train top3 : 
#     * data 크기가 340 * 10일때 더이상 overfitting을 해결하지 못하여 data 수를 늘렸다.
#     * data가 적을때는 train을 학습하는데 문제가 없었다.
#     * train set의 크기가 증가하여 validation accuracy가 2배 증가하였다.
#     * drop out을 convolution layer에도 적용했다.
#       * 적용한 layer가 많을수록 같은 epoch에 대해 train accuracy가 감소했다(1.0 -> 0.9로 감소).
#       * validation accuracy 증가했다.
#       * conv layer에 적용할 때 maxpool layer 다음에 적용했다.
#     * drop out으로 train accuracy가 감소하여 channel을 증가시켰다.
#       * train accuracy 0.90 -> 0.99로 증가, validation accuracy 0.52 -> 0.53
#       * convolution 연산은 receptive field와 fiter가 dot product한다. dot product는 선형 연산이다.
#       * 비록 activation function을 적용하여 non-linearity를 증가되지만 부족한 non-linearity를 channel을 증가하는 것으로 보충할 수 있다.
#       * train accuracy가 1.0에 도달하지 못한다는 것은 model의 capacity가 낮아서 data의 pattern을 학습할 수 없다는 뜻이다. 즉, non-linearity가 부족하다.
#     * l2 regularization 1e-3을 dense layer를 제외한 모든 layer에 적용하면 train accuracy와 validation accuracy 둘다 감소했다.
#       * regularization 제거
#     * epoch 50 -> 13으로 감소했다.
#       * epoch 50이면 train accuracy가 0.99에 도달하나 validation accuracy는 감소된다.
#       * 적절한 epoch으로 overfitting을 줄였다.
#     * overfitting 발생

# In[1]:


import cv2
import ast
import os
import gc
import json
import dask.bag
import random
import pathlib
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# In[12]:


IMG_HEIGHT = IMG_WIDTH = 64
SEED = 1818 # reproducible
SIZE_PER_CLASS = 1000 # 각 dataframe마다 SIZE개씩 불러옴. 총 사용할 데이터는 340 * SIZE
TRAIN_RATE, VAL_RATE, TEST_RATE = 0.64, 0.16, 0.2

# In[13]:


def stroke2arr(stroke, shape=(IMG_WIDTH, IMG_HEIGHT)):
    fig, ax = plt.subplots()
    for x, y in stroke:
        ax.plot(x, y, 'g', marker='.', linewidth=4)
        ax.axis('off')
    fig.canvas.draw()    
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    image = (cv2.resize(X, shape) / 255.)[::-1]
    return image[:, :, 1].reshape(1, *shape)

# In[14]:


# image size에 따라 선명함이 달라짐
# 32 64
# 128 256
# size는 64 혹은 128로 선택

df = pd.read_csv('../input/train_simplified/owl.csv', usecols=['drawing'], nrows=1)
arr = df['drawing'].apply(ast.literal_eval)[0]
shapes = ((32, 32), (64, 64), (128, 128), (256, 256))
fig=plt.figure(figsize=(8, 8))
columns, rows = 2, 2
for i in range(1, columns*rows +1):
    image = stroke2arr(arr, shapes[i-1])
    image = image.reshape(shapes[i-1])
    fig.add_subplot(rows, columns, i)
    plt.imshow(image, cmap='gray')
    plt.title('%dx%d' %shapes[i-1])
    plt.axis('off')

# In[15]:


def get_dataset(file_path_list):
    images = []
    labels = []
    for i, file_path in enumerate(tqdm.tqdm(file_path_list)):
        df = pd.read_csv(file_path,
                         usecols=['drawing'],
                         engine='python').sample(SIZE_PER_CLASS, random_state=SEED)
        arr = dask.bag.from_sequence(df.drawing.values).map(ast.literal_eval).map(stroke2arr)
        arr = np.array(arr.compute())
        images.append(arr)
#         arr = df['drawing'].apply(ast.literal_eval).apply(stroke2arr)
#         images.append(np.vstack(arr.values))
        labels += [i] * SIZE_PER_CLASS
    images = np.vstack(images).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    return images, labels

# In[16]:


# train data file path list
file_path_list = [*pathlib.Path('../input/train_simplified').glob('*')]
n_class = len(file_path_list) # class 수

# # file path에서 class name 추출 및 빈칸 -> "_"로 바꿈
class_name = np.char.array([
    *map(lambda file_path: file_path.name.split('.')[0].replace(' ', '_'),
         file_path_list)
    ])

# In[ ]:


print(images.shape)
print(len(labels))

# In[8]:


def split_data(images, labels):
    train_set, test_set, train_label, test_label = train_test_split(
            images,
            labels,
            test_size=TEST_RATE,
            shuffle=True,
            stratify=labels,
            random_state=SEED)
    train_set, val_set, train_label, val_label = train_test_split(
            train_set,
            train_label,
            test_size=VAL_RATE,
            shuffle=True,
            stratify=train_label,
            random_state=SEED)
    return train_set, val_set, test_set, train_label, val_label, test_label
def top_3(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)

# In[9]:


train_set, val_set, test_set, train_label, val_label, test_label = split_data(images, labels)

# In[11]:


BATCH_SIZE = 100
EPOCH = 13
reg_rate = 1e-3

tf.keras.backend.clear_session()

# os.environ['PYTHONHASHSEED'] = '0'                      
# np.random.seed(SEED)
# random.seed(SEED)
# tf.set_random_seed(SEED)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                               inter_op_parallelism_threads=1, 
#                               allow_soft_placement=True,
#                               device_count = {'CPU' : 1, 'GPU' : 1})
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# tf.keras.backend.set_session(sess)

InputLayer = tf.keras.layers.InputLayer
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
Dense = tf.keras.layers.Dense
initializer = tf.keras.initializers.glorot_normal
Flatten = tf.keras.layers.Flatten
BatchNorm = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
regularizer = tf.keras.regularizers.l2

model = tf.keras.Sequential()
model.add(
    InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 1))
)
model.add(
    BatchNorm()
)
model.add(
    Dropout(0.2)
)
model.add(
    Conv2D(16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')
)
model.add(
    BatchNorm()
)
model.add(
    MaxPool2D((2, 2), (2, 2), 'same')
)
model.add(
    Dropout(0.2)
)
model.add(
    Conv2D(32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')
)
model.add(
    BatchNorm()
)
model.add(
    MaxPool2D((2, 2), (2, 2), 'same')
)
model.add(
    Dropout(0.2)
)
model.add(
    Conv2D(64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')
)
model.add(
    BatchNorm()
)
model.add(
    MaxPool2D((2, 2), (2, 2), 'same')
)
model.add(
    Dropout(0.2)
)
model.add(
    Conv2D(128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')
)
model.add(
    BatchNorm()
)
model.add(
    MaxPool2D((2, 2), (2, 2), 'same')
)
model.add(
    Dropout(0.2)
)
model.add(
    Conv2D(256,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')
)
model.add(
    BatchNorm()
)
model.add(
    MaxPool2D((2, 2), (2, 2), 'same')
)
model.add(
    Flatten()
)
model.add(
    Dense(n_class,
          'softmax')
)
model.summary()
model.compile('adam', 'sparse_categorical_crossentropy', [top_3])
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_top_3',
#                                               patience=20,
#                                               min_delta=1e-4,
#                                               mode='max')
          train_label,\
          validation_data=(val_set, val_label),\
          epochs=EPOCH,\
          verbose=2,\
          batch_size=BATCH_SIZE)
#           callbacks=[early_stop],\

model.evaluate(test_set, test_label)

pd.DataFrame(data={'loss':model.history.history['loss'],
                   'val_loss':model.history.history['val_loss'],
                   'top_3':model.history.history['top_3'],
                   'val_top_3':model.history.history['val_top_3']}).plot(secondary_y=['top_3',
                                                                                      'val_top_3'])

random_index = np.arange(len(val_set))
np.random.shuffle(random_index)

val_pred = model.predict(val_set)
result = (-val_pred).argsort(axis=1)[:, :3]
val_label_pred = class_name[result]
val_label_actual = class_name[val_label]
val_text = val_label_pred[:, 0] + '\n' +\
            val_label_pred[:, 1] + '\n' +\
            val_label_pred[:, 2] + '\n' +\
            '['+val_label_actual+']'

columns, rows = 8, 8
fig = plt.figure(figsize=(16, 16))
for i in range(1, columns*rows +1):
    idx = random_index[i-1]
    arr = val_set[idx].reshape(IMG_WIDTH, IMG_HEIGHT)
    fig.add_subplot(rows, columns, i)
    plt.imshow(arr, cmap='gray')
    plt.title('%s' % val_text[idx])
    plt.axis('off')
    
plt.tight_layout()

# In[ ]:


gc.collect()
real_test = pd.read_csv('../input/test_simplified.csv',
                        usecols=['key_id', 'drawing'],
                        chunksize=2048)
real_test_images = []
for df in tqdm.tqdm(real_test, total=55):
    arr = dask.bag.from_sequence(df.drawing.values).map(ast.literal_eval).map(stroke2arr)
    arr = np.array(arr.compute())
    real_test_images.append(arr)
    del arr
real_test_images = np.vstack(real_test_images).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
real_test_images.shape

# In[ ]:


_test = pd.read_csv('../input/test_simplified.csv',
                    usecols=['key_id', 'drawing'], engine='python')
y_pred = model.predict(real_test_images)
result = (-y_pred).argsort(axis=1)[:, :3]
_class_name = np.char.array(class_name)
label_pred = _class_name[result]
submission = pd.DataFrame({'key_id':_test['key_id'],
                           'word': label_pred[:, 0] + ' ' + label_pred[:, 1] + ' ' + label_pred[:, 2]})
submission.to_csv('submission.csv', index=False)
