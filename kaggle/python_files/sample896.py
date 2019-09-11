#!/usr/bin/env python
# coding: utf-8

# # Kernel references 
# https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/output
# 
# Adding new features from Benjamin's great Kernel
# - Number of ads per user 
# - Average number of active days of all Ad's put up per user 
# - Average number of times all Ad's were made active per user 

# In[1]:


import pandas as pd 
import numpy as np 
import time 
import gc 

# In[2]:


np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model

from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
cores = 4

from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) 

# #  HANDLING VARIABLES 
#               
# *  Text variable -  title_description ( combination of title and description)
#   Using title and description fields separately gives a small boost to local CV and LB score but takes twice as long to train
# 
# *  Categorical variables - region, city, category_name, parent_category_name, image_code_1, param_1, param123
#    param_1, param_2, param_3 variables are combined since param_2 and param_3 have close to 50% NaN values. 
# 
# *  Continous variables - price, item_seq_number 
# 
# *  Continuous variables based on train_active -  avg_days_up_user, avg_times_up_user, n_user_items
# 
#     These features are from Benjamin's great Kernel 
#     https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/output
# 
# The Following features are uploaded as a data set in a csv aggregated_features generated from Benjamin's kernel mentioned above. 
# 
# - Number of ads per user 
# - Average number of active days of all Ad's put up per user 
# - Average number of times all Ad's were made active per user 
# 
# 
# 
#    

# # STEP BY STEP EXPLANATION 
# # PSEUDO CODE FOR WORK FLOW 
# 
# Step 1 :  Preprocess input data. Generate the target array 
# 
# Step 2:  Feature engineering with  tokens for word sequences and label encoding for categorical variables 
# 
# Step 3:  Transform train using word tokens and label encoders
# 
# Step 4:   Generate word embedding file ( In this case Fast Text is used )
# 
# Step 5:  Initialize the RNN model with learning rates, epochs, batch size and any necessary parameters 
# 
# Step 6:  Run a KFold on training data and the target variable. For each fold call the prediction function to 
#              generate the output 'deal_probability'. 
#              
# Step 7:  Combine the 'deal_probability' values in any number of ways 
#             Simple Average,  Log Average, Output with least RMSE etc to generate final output 

# # FUNCTION - preprocess_dataset
#  
# *  Handling Missing Values
# *  Casting categorical variables into type "category"
# *  Combine 3 param variables into single feature
# 
# # FUNCTION - keras_fit 
# 
# * Combine title and description into a single column and delete title and description 
# 
# TOKENIZATION OF TEXT 
# 
# *    Tokenize the sentences in 'title_description'. Maximum number of words in vocabulary is taken as 200,000.
# *    The number was chosen based on a quick look at the corresponding word_counts(). 
# *    Words with index greater than 200,000 appear only less than 4 times. 
#    
# LABEL ENCODING OF CATEGORICAL VARIABLES - 
# 
# *  sklearn label encoding is used 
# *  The label encoding is done for train and test values to ensure that no new labels are encountered during the prediction 
#    phase on test
# *  A new Label Encoder is created for every categorical field that is to be label encoded 
# *  The output of this function will generate the transformed train data frame along with the word tokenizer and label encoder labels 
# 
# LOG TRANSFORMATION OF CONTINUOUS VARIABLES - 
# 
# * np.log1p is applied on the continuous values (log1p to avoid log of zero issues)
# * Log transformation is used on price, item_seq_number
# * Log transformation is used on the following values of avg_days_up_user, avg_times_up_user, n_user_items
# 
# OUTPUT OF FUNCTION 
# 
# *  train data frame after necessary feature engineering 
# *  Tokenizer for word features and label encoders for each of the categorical variables  
# 
# # FUNCTION  keras_train_transform 
# 
# * Apply the tokenizer and label encoders to the vectorized train data frame 
# 
# Label Encoding - Assigns a unique label name to each category type . This is the transformation process. 
# Text sequences - Each word in a sentence is assigned a unique label name. The seq_title_description field is generated 
#                               which is a list of all the indexes for the words in the sentence. If a word index is greater than the limit 
#                               provided by us ( 200,000 in this case) the word is not included in the sequence. 
# 
# * Input - train data frame with raw variables 
# * Output - train data frame with transformed variables ready for training 
# 
# # FUNCTION  keras_train_transform 
# 
# * Apply the tokenizer and label encoders to the vectorized test data frame 
# * Input - test data frame with raw variables 
# * Output - test data frame with transformed variables ready for prediction 
#    
#  # FUNCTION  get_keras_data
#  
# * It is easy to pass an input to the RNN fit call in Keras when Training and Validation data sets are passed as Dictionaries 
# * Converts the transformed train data frame into a dictionary 
# *  The function also takes the list of word indexes and pads it with zeros to a pre determined length 
# * Each key in the dictionary will be the corresponding column name 
#    

# In[3]:


def preprocess_dataset(dataset):
    
    t1 = time.time()
    print("Filling Missing Values.....")
    
    dataset['price'] = dataset['price'].fillna(0).astype('float32')
    dataset['param_1'].fillna(value='missing', inplace=True)
    dataset['param_2'].fillna(value='missing', inplace=True)
    dataset['param_3'].fillna(value='missing', inplace=True)
    
    dataset['param_1'] = dataset['param_1'].astype(str)
    dataset['param_2'] = dataset['param_2'].astype(str)
    dataset['param_3'] = dataset['param_3'].astype(str)
    
    print("Casting data types to type Category.......")
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['parent_category_name'] = dataset['parent_category_name'].astype('category')
    dataset['region'] = dataset['region'].astype('category')
    dataset['city'] = dataset['city'].astype('category')
    
    dataset['image_top_1'] = dataset['image_top_1'].fillna('missing')
    dataset['image_code'] = dataset['image_top_1'].astype('str')
    del dataset['image_top_1']
    gc.collect()

    #dataset['week'] = pd.to_datetime(dataset['activation_date']).dt.week.astype('uint8')
    #dataset['day'] = pd.to_datetime(dataset['activation_date']).dt.day.astype('uint8')
    #dataset['wday'] = pd.to_datetime(dataset['activation_date']).dt.dayofweek.astype('uint8')
    del dataset['activation_date']
    gc.collect()
    
    print("Creating New Feature.....")
    dataset['param123'] = (dataset['param_1']+'_'+dataset['param_2']+'_'+dataset['param_3']).astype(str)
    del dataset['param_2'], dataset['param_3']
    gc.collect()
        
    print("PreProcessing Function completed.")
    
    return dataset

def keras_fit(train):
    
    t1 = time.time()
    train['title_description']= (train['title']+" "+train['description']).astype(str)
    del train['description'], train['title']
    gc.collect()
    
    print("Start Tokenization.....")
    tokenizer = text.Tokenizer(num_words = max_words_title_description)
    all_text = np.hstack([train['title_description'].str.lower()])
    tokenizer.fit_on_texts(all_text)
    del all_text
    gc.collect()
    
    print("Loading Test for Label Encoding on Train + Test")
    use_cols_test = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'image_top_1', 'activation_date']
    test = pd.read_csv("../input/avito-demand-prediction/test.csv", usecols = use_cols_test)
    
    test['image_top_1'] = test['image_top_1'].fillna('missing')
    test['image_code'] = test['image_top_1'].astype('str')
    del test['image_top_1']
    gc.collect()
    
    #test['week'] = pd.to_datetime(test['activation_date']).dt.week.astype('uint8')
    #test['day'] = pd.to_datetime(test['activation_date']).dt.day.astype('uint8')
    #test['wday'] = pd.to_datetime(test['activation_date']).dt.dayofweek.astype('uint8')
    del test['activation_date']
    gc.collect()
    
    test['param_1'].fillna(value='missing', inplace=True)
    test['param_1'] = test['param_1'].astype(str)
    test['param_2'].fillna(value='missing', inplace=True)
    test['param_2'] = test['param_2'].astype(str)
    test['param_3'].fillna(value='missing', inplace=True)
    test['param_3'] = test['param_3'].astype(str)

    print("Creating New Feature.....")
    test['param123'] = (test['param_1']+'_'+test['param_2']+'_'+test['param_3']).astype(str)
    del test['param_2'], test['param_3']
    gc.collect()
    
    ntrain = train.shape[0]
    DF = pd.concat([train, test], axis = 0)
    del train, test
    gc.collect()
    print(DF.shape)
    
    print("Start Label Encoding process....")
    le_region = LabelEncoder()
    le_region.fit(DF.region)
    
    le_city = LabelEncoder()
    le_city.fit(DF.city)
    
    le_category_name = LabelEncoder()
    le_category_name.fit(DF.category_name)
    
    le_parent_category_name = LabelEncoder()
    le_parent_category_name.fit(DF.parent_category_name)
    
    le_param_1 = LabelEncoder()
    le_param_1.fit(DF.param_1)
    
    le_param123 = LabelEncoder()
    le_param123.fit(DF.param123)
    
    le_image_code = LabelEncoder()
    le_image_code.fit(DF.image_code)
    
    #le_week = LabelEncoder()
    #le_week.fit(DF.week)
    #le_day = LabelEncoder()
    #le_day.fit(DF.day)
    #le_wday = LabelEncoder()
    #le_wday.fit(DF.wday)
    
    train = DF[0:ntrain]
    del DF 
    gc.collect()
    
    train['price'] = np.log1p(train['price'])
    train['avg_days_up_user'] = np.log1p(train['avg_days_up_user'])
    train['avg_times_up_user'] = np.log1p(train['avg_times_up_user'])
    train['n_user_items'] = np.log1p(train['n_user_items'])
    train['item_seq_number'] = np.log(train['item_seq_number'])
    print("Fit on Train Function completed.")
    
    return train, tokenizer, le_region, le_city, le_category_name, le_parent_category_name, le_param_1, le_param123, le_image_code

def keras_train_transform(dataset):
    
    t1 = time.time()
    
    dataset['seq_title_description']= tokenizer.texts_to_sequences(dataset.title_description.str.lower())
    print("Transform done for test")
    print("Time taken for Sequence Tokens is"+str(time.time()-t1))
    del train['title_description']
    gc.collect()

    dataset['region'] = le_region.transform(dataset['region'])
    dataset['city'] = le_city.transform(dataset['city'])
    dataset['category_name'] = le_category_name.transform(dataset['category_name'])
    dataset['parent_category_name'] = le_parent_category_name.transform(dataset['parent_category_name'])
    dataset['param_1'] = le_param_1.transform(dataset['param_1'])
    dataset['param123'] = le_param123.transform(dataset['param123'])
    #dataset['day'] = le_day.transform(dataset['day'])
    #dataset['week'] = le_week.transform(dataset['week'])
    #dataset['wday'] = le_wday.transform(dataset['wday'])
    dataset['image_code'] = le_image_code.transform(dataset['image_code'])
    
    print("Transform on test function completed.")
    
    return dataset
    
def keras_test_transform(dataset):
    
    t1 = time.time()
    dataset['title_description']= (dataset['title']+" "+dataset['description']).astype(str)
    del dataset['description'], dataset['title']
    gc.collect()
    
    dataset['seq_title_description']= tokenizer.texts_to_sequences(dataset.title_description.str.lower())
    print("Transform done for test")
    print("Time taken for Sequence Tokens is"+str(time.time()-t1))
    
    del dataset['title_description']
    gc.collect()

    dataset['region'] = le_region.transform(dataset['region'])
    dataset['city'] = le_city.transform(dataset['city'])
    dataset['category_name'] = le_category_name.transform(dataset['category_name'])
    dataset['parent_category_name'] = le_parent_category_name.transform(dataset['parent_category_name'])
    dataset['param_1'] = le_param_1.transform(dataset['param_1'])
    dataset['param123'] = le_param123.transform(dataset['param123'])
    #dataset['day'] = le_day.transform(dataset['day'])
    #dataset['week'] = le_week.transform(dataset['week'])
    #dataset['wday'] = le_wday.transform(dataset['wday'])
    dataset['image_code'] = le_image_code.transform(dataset['image_code'])
    
    dataset['price'] = np.log1p(dataset['price'])
    dataset['item_seq_number'] = np.log(dataset['item_seq_number'])
    dataset['avg_days_up_user'] = np.log1p(dataset['avg_days_up_user'])
    dataset['avg_times_up_user'] = np.log1p(dataset['avg_times_up_user'])
    dataset['n_user_items'] = np.log1p(dataset['n_user_items'])
    
    print("Transform on test function completed.")
    
    return dataset
    
def get_keras_data(dataset):
    X = {
        'seq_title_description': pad_sequences(dataset.seq_title_description, maxlen=max_seq_title_description_length)
        ,'region': np.array(dataset.region)
        ,'city': np.array(dataset.city)
        ,'category_name': np.array(dataset.category_name)
        ,'parent_category_name': np.array(dataset.parent_category_name)
        ,'param_1': np.array(dataset.param_1)
        ,'param123': np.array(dataset.param123)
        ,'image_code':np.array(dataset.image_code)
        ,'avg_ad_days':np.array(dataset.avg_days_up_user )
        ,'avg_ad_times':np.array(dataset.avg_times_up_user)
        ,'n_user_items':np.array(dataset.n_user_items)
        ,'price': np.array(dataset[["price"]])
        ,'item_seq_number': np.array(dataset[["item_seq_number"]])
    }
    
    print("Data ready for Vectorization")
    
    return X


# In[4]:


# Loading Train data - No Params, No Image data 
dtypes_train = {
                'price': 'float32',
                'deal probability': 'float32',
                'item_seq_number': 'uint32'
}

# No user_id
use_cols = ['item_id', 'user_id', 'image_top_1', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price', 'item_seq_number', 'activation_date', 'deal_probability']
train = pd.read_csv("../input/avito-demand-prediction/train.csv", parse_dates=["activation_date"], usecols = use_cols, dtype = dtypes_train)

train_features = pd.read_csv('../input/avito-additional-features/aggregated_features.csv')
train = train.merge(train_features, on = ['user_id'], how = 'left')
del train_features
gc.collect()

train['avg_days_up_user'] = train['avg_days_up_user'].fillna(0).astype('uint32')
train['avg_times_up_user'] = train['avg_times_up_user'].fillna(0).astype('uint32')
train['n_user_items'] = train['n_user_items'].fillna(0).astype('uint32')

y_train = np.array(train['deal_probability'])

del train['deal_probability']
gc.collect()

max_seq_title_description_length = 100
max_words_title_description = 200000

train = preprocess_dataset(train)
train, tokenizer, le_region, le_city, le_category_name, le_parent_category_name, le_param_1, le_param123, le_image_code = keras_fit(train)
train = keras_train_transform(train)
print("Tokenization done and TRAIN READY FOR Validation splitting")

# Calculation of max values for Categorical fields 

max_region = np.max(train.region.max())+2
max_city= np.max(train.city.max())+2
max_category_name = np.max(train.category_name.max())+2
max_parent_category_name = np.max(train.parent_category_name.max())+2
max_param_1 = np.max(train.param_1.max())+2
max_param123 = np.max(train.param123.max())+2
#max_week = np.max(train.week.max())+2
#max_day = np.max(train.day.max())+2
#max_wday = np.max(train.wday.max())+2
max_image_code = np.max(train.image_code.max())+2


del train['item_id'], train['user_id']
gc.collect()


# # EMBEDDING FILE - FASTTEXT 
# 
# * A 300 Dimension Fast Text Vector is used 
# * Calculating the vocab_size from tokenizer is very important. The values are used in the Embedding in the RNNmodel. 
# * For every word available in the data set the corresponding 300 D vector is added to the embedding matrix 
#     - The embedding matrix is of size vocab_size, embedding size which in this case is 300. 
#     - The idea is that the index of a word in the vocabulary also becomes the row number for that word in the matrix 
#     - This way we can easily retrieve the word vector of any word 
# * For words not available in the data set the vector value is assigned as zero. 
# * One can also try replacing the zero vectors with average of available vectors or a random vector using np.random
# * We keep track of the number of words in the dataset for which word vectors are available 
# 
# HOW TO USE THESE EMBEDDINGS 
# 
# The embedding layer can be used in two ways 
# 
# Trainable = True :  The initial word vector values are trained further in the RNN training stage. Takes a very long time 
# Trainable = False:  The parameters for title_description are not trained and are kept at the same value through all the epochs of the RNN training process. 

# In[5]:


# EMBEDDINGS COMBINATION 
# FASTTEXT

EMBEDDING_DIM1 = 300
EMBEDDING_FILE1 = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

vocab_size = len(tokenizer.word_index)+2
EMBEDDING_DIM1 = 300# this is from the pretrained vectors
embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
print(embedding_matrix1.shape)
# Creating Embedding matrix 
c = 0 
c1 = 0 
w_Y = []
w_No = []
for word, i in tokenizer.word_index.items():
    if word in embeddings_index1:
        c +=1
        embedding_vector = embeddings_index1[word]
        w_Y.append(word)
    else:
        embedding_vector = None
        w_No.append(word)
        c1 +=1
    if embedding_vector is not None:    
        embedding_matrix1[i] = embedding_vector

print(c,c1, len(w_No), len(w_Y))
print(embedding_matrix1.shape)
del embeddings_index1
gc.collect()

print(" FAST TEXT DONE")


# In[6]:


print(vocab_size)

# # GENERATE RNN MODEL 
# 
# * Initialize the inputs for all the variables. The word sequences for title_description have a length of 100 and other variables have a length of 1 
# * An embedding layer is generated for each of the Categorical and Text variables 
# * NO Embeddings are required for continous variable like price 
# * A Recurrent Neural network of 50 GRU units is applied to the embedding of title_description 
# * The embeddings of the categorical variables are Flattened using the Flatten() command 
# *  The GRU, Flatten values are concatenated with the continuous values and treated as the main layer 
# *  This main layer is then passed through 2 Dense layers of 512 layers and 64 layers 
#   
# WHAT CAN BE TUNED 
# 
# * Architecture of the network itself.  Bidirectional GRU's with Batch Normalization is worth a shot. 
# * Learning rate 
# * Number of Dense layers and the number of hidden units in each layer 
# * Dropout values 
# * The optimizer function - Adam is currently used. Other options are available 
# * Number of GRU units 
# 
# 
# 
# 

# In[7]:


def RNN_model():

    #Inputs
    seq_title_description = Input(shape=[100], name="seq_title_description")
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    param_1 = Input(shape=[1], name="param_1")
    param123 = Input(shape=[1], name="param123")
    image_code = Input(shape=[1], name="image_code")
    price = Input(shape=[1], name="price")
    item_seq_number = Input(shape = [1], name = 'item_seq_number')
    avg_ad_days = Input(shape=[1], name="avg_ad_days")
    avg_ad_times = Input(shape=[1], name="avg_ad_times")
    n_user_items = Input(shape=[1], name="n_user_items")
    
    #Embeddings layers

    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title_description)
    emb_region = Embedding(vocab_size, 10)(region)
    emb_city = Embedding(vocab_size, 10)(city)
    emb_category_name = Embedding(vocab_size, 10)(category_name)
    emb_parent_category_name = Embedding(vocab_size, 10)(parent_category_name)
    emb_param_1 = Embedding(vocab_size, 10)(param_1)
    emb_param123 = Embedding(vocab_size, 10)(param123)
    emb_image_code = Embedding(vocab_size, 10)(image_code)

    rnn_layer1 = GRU(50) (emb_seq_title_description)
    
    #main layer
    main_l = concatenate([
          rnn_layer1
        , Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_category_name)
        , Flatten() (emb_parent_category_name)
        , Flatten() (emb_param_1)
        , Flatten() (emb_param123)
        , Flatten() (emb_image_code)
        , avg_ad_days
        , avg_ad_times
        , n_user_items
        , price
        , item_seq_number
    ])
    
    main_l = Dropout(0.1)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="sigmoid") (main_l)
    
    #model
    model = Model([seq_title_description, region, city, category_name, parent_category_name, param_1, param123, price, item_seq_number, image_code, avg_ad_days, avg_ad_times, n_user_items], output)
    model.compile(optimizer = 'adam',
                  loss= root_mean_squared_error,
                  metrics = [root_mean_squared_error])
    return model

def rmse(y, y_pred):

    Rsum = np.sum((y - y_pred)**2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum/n)
    return RMSE 

def eval_model(model, X_test1):
    val_preds = model.predict(X_test1)
    y_pred = val_preds[:, 0]
    
    y_true = np.array(y_test1)
    
    yt = pd.DataFrame(y_true)
    yp = pd.DataFrame(y_pred)
    
    print(yt.isnull().any())
    print(yp.isnull().any())
    
    v_rmse = rmse(y_true, y_pred)
    print(" RMSE for VALIDATION SET: "+str(v_rmse))
    return v_rmse

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

# # PREDICTING THE OUTPUT 
# 
# * A function is defined with parameter as the trained modelRNN
# * Test data is loaded in chunks to AVOID MEMORY OVERLOAD 
# * Test data is preprocessed 
# * Transformations are applied using Keras word token, label encoders and the train_active based features 
# * Generate dictionary for Test 
# * Predict the output 

# In[ ]:


def predictions(model):
    import time
    t1 = time.time()
    def load_test():
        for df in pd.read_csv('../input/avito-demand-prediction/test.csv', chunksize= 250000):
            yield df

    item_ids = np.array([], dtype=np.int32)
    preds= np.array([], dtype=np.float32)

    i = 0 
    
    for df in load_test():
    
        i +=1
        print(df.dtypes)
        item_id = df['item_id']
        print(" Chunk number is "+str(i))
    
        test = preprocess_dataset(df)
    
        train_features = pd.read_csv('../input/avito-additional-features/aggregated_features.csv')
        test = test.merge(train_features, on = ['user_id'], how = 'left')
        del train_features
        gc.collect()
    
        print(test.dtypes)
        
        test['avg_days_up_user'] = test['avg_days_up_user'].fillna(0).astype('uint32')
        test['avg_times_up_user'] = test['avg_times_up_user'].fillna(0).astype('uint32')
        test['n_user_items'] = test['n_user_items'].fillna(0).astype('uint32')
        test = keras_test_transform(test)
        del df
        gc.collect()
    
        print(test.dtypes)
    
        X_test = get_keras_data(test)
        del test 
        gc.collect()
    
        Batch_Size = 512*3
        preds1 = modelRNN.predict(X_test, batch_size = Batch_Size, verbose = 1)
        print(preds1.shape)
        del X_test
        gc.collect()
        print("RNN Prediction is done")

        preds1 = preds1.reshape(-1,1)
        #print(predsl.shape)
        preds1 = np.clip(preds1, 0, 1)
        print(preds1.shape)
        item_ids = np.append(item_ids, item_id)
        print(item_ids.shape)
        preds = np.append(preds, preds1)
        print(preds.shape)
        
    print("All chunks done")
    t2 = time.time()
    print("Total time for Parallel Batch Prediction is "+str(t2-t1))
    return preds 

# # PROCESSING FOR KFOLD 
# 
# * K Fold accepts only arrays as inputs 
# * The train data frame is converted into array to generate the indexes for KFOLD 
# * The array is then used to reconstruct a data frame with which a dictionary is generated 
# * Dictionary is the best input type for model.fit in Keras 

# In[ ]:


train1 = np.array(train.values)
del train
gc.collect()

def get_data_frame(dataset):
    
    DF = pd.DataFrame()
    
    DF['avg_days_up_user'] = np.array(dataset[:,0])
    DF['avg_times_up_user'] = np.array(dataset[:,1])
    DF['category_name'] = np.array(dataset[:,2])
    DF['city'] = np.array(dataset[:,3])
    DF['image_code'] = np.array(dataset[:,4])
    DF['item_seq_number'] = np.array(dataset[:,5])
    DF['n_user_items'] = np.array(dataset[:,6])
    DF['param123'] = np.array(dataset[:,7])
    DF['param_1'] = np.array(dataset[:,8])
    DF['parent_category_name'] = np.array(dataset[:,9])
    DF['price'] = np.array(dataset[:,10])
    DF['region'] = np.array(dataset[:,11])
    DF['seq_title_description'] = np.array(dataset[:,12])
    
    return DF 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time 
skf = KFold(n_splits = 3)
Kfold_preds_final = []
k = 0
RMSE = []

for train_idx, test_idx in skf.split(train1, y_train):
    
    print("Number of Folds.."+str(k+1))
    
    # Initialize a new Model for Current FOLD 
    epochs = 1
    batch_size = 512 * 3
    steps = (int(train1.shape[0]/batch_size))*epochs
    lr_init, lr_fin = 0.009, 0.0045
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    K.set_value(modelRNN.optimizer.lr, lr_init)
    K.set_value(modelRNN.optimizer.decay, lr_decay)

    #K Fold Split 
    
    X_train1, X_test1 = train1[train_idx], train1[test_idx]
    print(X_train1.shape, X_test1.shape)
    y_train1, y_test1 = y_train[train_idx], y_train[test_idx]
    print(y_train1.shape, y_test1.shape)
    gc.collect()
    
    print(type(X_train1))
    print(X_train1.shape)
    print(type(X_train1[:,12]))
    
    X_train_final = get_data_frame(X_train1)
    X_test_final = get_data_frame(X_test1)
    
    del X_train1, X_test1
    gc.collect()
    
    X_train_f = get_keras_data(X_train_final)
    X_test_f = get_keras_data(X_test_final)
    
    del X_train_final, X_test_final
    gc.collect()

    # Fit the NN Model 
    for i in range(3):
        hist = modelRNN.fit(X_train_f, y_train1, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_test_f, y_test1), verbose=1)

    del X_train_f
    gc.collect()

    # Print RMSE for Validation set for Kth Fold 
    v_rmse = eval_model(modelRNN, X_test_f)
    RMSE.append(v_rmse)
    
    del X_test_f
    del y_train1, y_test1
    gc.collect()
    
    # Predict test set for Kth Fold 
    preds = predictions(modelRNN)
    del modelRNN 
    gc.collect()

    print("Predictions done for Fold "+str(k))
    print(preds.shape)
    Kfold_preds_final.append(preds)
    del preds
    gc.collect()
    print("Number of folds completed...."+str(len(Kfold_preds_final)))
    print(Kfold_preds_final[k][0:10])

print("All Folds completed"+str(k+1))   
print("RNN FOLD MODEL Done")

# # SELECTING KFOLD OUTPUT 
# 
# *  Average of all Kfold predictions is catpured in pred_final1 
# *    The KFOLD run with least RMSE score is identified and the corresponding output is taken as pred_final2 
# *    2 Separate output files are generated for comparison 

# In[ ]:


pred_final1 = np.average(Kfold_preds_final, axis =0) # Average of all K Folds
print(pred_final1.shape)

min_value = min(RMSE)
RMSE_idx = RMSE.index(min_value)
print(RMSE_idx)
pred_final2 = Kfold_preds_final[RMSE_idx]
print(pred_final2.shape)

#del Kfold_preds_final, train1
gc.collect()

# In[ ]:


pred_final1[0:5]

# In[ ]:


pred_final2[0:5]

# In[ ]:


test_cols = ['item_id']
test = pd.read_csv('../input/avito-demand-prediction/test.csv', usecols = test_cols)

# using Average of KFOLD preds 

submission1 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission1['item_id'] = test['item_id']
submission1['deal_probability'] = pred_final1

print("Check Submission NOW!!!!!!!!@")
submission1.to_csv("Avito_Shanth_RNN_AVERAGE.csv", index=False)


# In[ ]:


# Using KFOLD preds with Minimum value 
submission2 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission2['item_id'] = test['item_id']
submission2['deal_probability'] = pred_final2

print("Check Submission NOW!!!!!!!!@")
submission2.to_csv("Avito_Shanth_RNN_MIN.csv", index=False)

# In[ ]:



