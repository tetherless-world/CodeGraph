#!/usr/bin/env python
# coding: utf-8

# CNN classifier for handwritten digits of the MNIST dataset. The dataset consists of 42000 images of size 28x28 = 784 pixels (one color number) including the corresponding labels from 0,..,9. The basic architecture of the NN is given by:
# 
# - Layer: input = [42000,784]
# - Layer: Conv1 -> ReLu -> MaxPool: [.,14,14,32] 
# - Layer: Conv2 -> ReLu -> MaxPool: [.,7,7,64]
# - Layer: FC -> ReLu: [.,1024]
# - Layer: FC -> ReLu: [.,10]
# 
# Using a split of 95%/5% on the labeled data this implementation, trained on 40000 training images for 8 epochs with suitable hyperparameters, achieves a 99.45% accuracy on the validation set of 2000 images.
# 
# ## Libraries and Settings

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # cm = colormap
import tensorflow as tf
import os;
import itertools

dir_logs = os.getcwd()+'/logs'; # directory to save models
val_set_size = 2000; # validation set size 

#display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));

# ## Data Preprocessing

# In[ ]:


## read training and validation data [42000,785] dataframe

if os.path.isfile('../input/train.csv'):
    data = pd.read_csv('../input/train.csv') # on kaggle 
    print('train.csv loaded: data({0[0]},{0[1]})'.format(data.shape))
elif os.path.isfile('data/train.csv'):
    data = pd.read_csv('data/train.csv') # on local environment
    print('train.csv loaded: data({0[0]},{0[1]})'.format(data.shape))
else:
    print('Error: train.csv not found')

# In[ ]:


## look at data and split into training and validation sets

# extract images
images = data.iloc[:,1:].values # (42000,784) array
images = images.astype(np.float) # convert from int64 to float
images = np.multiply(images, 1.0 / 255.0) # convert from [0:255] to [0.0:1.0]
image_size = images.shape[1] # = 784
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8) # = 28

# extract image labels
labels_flat = data.iloc[:,0].values
labels_count = np.unique(labels_flat).shape[0]; # number of different labels = 10

#plot some images and labels
plt.figure(figsize=(15,2))
for i in range(0,10):
    plt.subplot(2,10,1+i)
    plt.title(labels_flat[i])
    plt.imshow(images[i].reshape(image_width,image_height),cmap=cm.binary)
    
# convert class labels from scalars to one-hot vectors e.g. 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# labels in one hot representation
labels = dense_to_one_hot(labels_flat, labels_count).astype(np.uint8)
#labels = labels.astype(np.uint8)

# split data into training & validation
train_images = images[val_set_size:]
train_labels = labels[val_set_size:]
val_images = images[:val_set_size]
val_labels = labels[:val_set_size]

print('images({0[0]},{0[1]}),'.format(images.shape),'labels_flat({0[0]})'.format(labels_flat.shape))
print('train_images({0[0]},{0[1]})'.format(train_images.shape),'labels({0[0]},{0[1]})'.format(labels.shape),'val_images({0[0]},{0[1]})'.format(val_images.shape),'val_labels({0[0]},{0[1]})'.format(val_labels.shape))
print ('image_size = {0}, image_width = {1}, image_height = {2}, labels_count = {3}'.format(image_size,image_width,image_height,labels_count))


# ## TensorFlow Graph

# In[ ]:


#tf.set_random_seed(1)
#np.random.seed(1)

# weight and bias initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) #  positive bias
    return tf.Variable(initial)

# 2D convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# variables for input and output 
x = tf.placeholder('float', shape=[None, image_size])
y_ = tf.placeholder('float', shape=[None, labels_count])

# 1. layer: convolution + max pooling
image = tf.reshape(x, [-1,28,28,1]) # (.,784) => (.,28,28,1)
W_conv1 = weight_variable([5, 5, 1, 32]) # (5,5,1,32)
b_conv1 = bias_variable([32]) # (32)
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1) # => (.,28,28,32)
h_pool1 = max_pool_2x2(h_conv1) # => (.,14,14,32)

# 2. layer: convolution + max pooling
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # => (.,14,14,64)
h_pool2 = max_pool_2x2(h_conv2) # => (.,7,7,64)

# 3.layer: fully connected
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) # (.,7,7,64) => (.,3136)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # => (.,1024)

# dropout
tf_keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, tf_keep_prob)

# 4.layer: fully connected
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # => (.,10)

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# optimisation function
global_step = tf.Variable(0, trainable=False)
tf_learn_rate = tf.placeholder(dtype='float', name="tf_learn_rate")
train_step = tf.train.AdamOptimizer(tf_learn_rate).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# prediction function
predict = tf.argmax(tf.nn.softmax(y),1) # [0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1

# function: to get the next mini batch
def next_batch(batch_size):
    global train_images, train_labels, index_in_epoch;
    assert batch_size <= num_examples
 
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > num_examples:
        perm = np.arange(num_examples) 
        np.random.shuffle(perm) # shuffle the data
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# ## Training and Validation

# In[ ]:


## set parameters

sess = tf.InteractiveSession() # start TensorFlow session
sess.run(tf.global_variables_initializer()) # initialize global variables

# variables and parameters
num_examples = train_images.shape[0];
index_in_epoch = 0;
train_acc, val_acc, train_loss, val_loss = np.array([]),np.array([]),np.array([]),np.array([]);  
log_step = 50; # log results each step
epoch_no = 8; # no of epochs 

# test hyperparameters
mb_size_range = [50]; # mini batch size
keep_prob_range = [0.5]; # dropout regularization with keeping probability
learn_rate_range = [10*1e-4, 5*1e-4, 2.5*1e-4, 1*1e-4, 0.5*1e-4, 0.25*1e-4, 0.1*1e-4, 
                    0.05*1e-4, 0.025*1e-4, 0.01*1e-4];
learn_rate_step = 1.0; # change learning rate each learn_rate_step in epochs

# In[ ]:


## training model

for mb_size,keep_prob in itertools.product(mb_size_range,keep_prob_range):
    mb_no = int(np.floor(epoch_no*num_examples/mb_size)); # no of mini batches
    learn_rate_step = int(np.floor(learn_rate_step*num_examples/mb_size)); # steps in batches
    print('epoch_no = %.0f, mb_size = %.0f, keep_prob = %.2f'%(epoch_no,mb_size,keep_prob))
    learn_rate_pos = -1;
    
    for i in range(0,mb_no+1):
        
        if (i%learn_rate_step == 0) and ((learn_rate_pos+1) < len(learn_rate_range)):
            learn_rate_pos+=1;
            learn_rate = learn_rate_range[learn_rate_pos]  # adapt learn_rate
            print('set current learn rate to: %.6f'%learn_rate)
        
        #learn_rate = 0.001*1e-4;
        
        batch_xs, batch_ys = next_batch(mb_size) #get new batch
        
        if i > 0:
             sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, 
                                                tf_keep_prob: keep_prob, 
                                                tf_learn_rate: learn_rate})
        if i%log_step == 0 or i == mb_no:
            train_loss = np.append(train_loss, sess.run(cross_entropy, feed_dict={x:train_images[0:2000], y_:train_labels[0:2000], tf_keep_prob:1.0}));
            train_acc = np.append(train_acc, accuracy.eval(feed_dict={x:train_images[0:2000], y_:train_labels[0:2000], tf_keep_prob:1.0}));      
            if val_set_size > 0:
                train_loss = np.append(train_loss, sess.run(cross_entropy, feed_dict={x:train_images[0:val_set_size], y_:train_labels[0:val_set_size], tf_keep_prob:1.0}));
                train_acc = np.append(train_acc, accuracy.eval(feed_dict={x:train_images[0:val_set_size], y_:train_labels[0:val_set_size], tf_keep_prob:1.0}));      
                val_loss = np.append(val_loss, sess.run(cross_entropy, feed_dict={x:val_images, y_: val_labels, tf_keep_prob: 1.0}));
                val_acc = np.append(val_acc, accuracy.eval(feed_dict={x: val_images, y_: val_labels,tf_keep_prob: 1.0}));                                  
            else: 
                val_loss = [0]; val_acc = [0];
            print('%.2f epoch: train/val loss = %.4f/%.4f , train/val acc = %.4f/%.4f'%(i*mb_size/num_examples,train_loss[-1],val_loss[-1],train_acc[-1], val_acc[-1]))

    # save model
    #if not os.path.exists(dir_logs): # check if directory for logs exists
    #    os.makedirs(dir_logs)
    #np.savez(dir_logs+'/model.npz', 
    #        learn_rate = learn_rate, keep_prob = keep_prob, mb_size = mb_size, log_step = log_step,
    #        W_conv1 = np.asarray(W_conv1.eval()), b_conv1 = np.asarray(b_conv1.eval()), W_conv2 = np.asarray(W_conv2.eval()),
    #        b_conv2 = np.asarray(b_conv2.eval()), W_fc1 = np.asarray(W_fc1.eval()), b_fc1 = np.asarray(b_fc1.eval()),
    #        W_fc2 = np.asarray(W_fc2.eval()), b_fc2 = np.asarray(b_fc2.eval()),
    #        train_loss = train_loss, val_loss = val_loss, train_acc = train_acc,
    #        val_acc = val_acc, val_loss_final = val_loss_final, val_acc_final = val_acc_final);

    #close session
    #sess.close();


# In[ ]:


'''
## load model

#print(dir_logs + ': ' + str(os.listdir(dir_logs)))
print('load '+ dir_logs + '/model.npz')
npzFile = np.load(dir_logs+'/model.npz');
#print(npzFile.files);
learn_rate = npzFile['learn_rate'];
keep_prob = npzFile['keep_prob'];
mb_size = npzFile['mb_size'];
log_step = npzFile['log_step'];
train_loss = npzFile['train_loss'];
val_loss = npzFile['val_loss'];
train_acc = npzFile['train_acc'];
val_acc = npzFile['val_acc'];
val_loss_final = npzFile['val_loss_final'];
val_acc_final = npzFile['val_acc_final'];

sess = tf.InteractiveSession() # start TensorFlow session
#sess.run(tf.global_variables_initializer()) # initialiue global variables
W_conv1.load(npzFile['W_conv1'], session=sess)
b_conv1.load(npzFile['b_conv1'], session=sess)
W_conv2.load(npzFile['W_conv2'], session=sess)
b_conv2.load(npzFile['b_conv2'], session=sess)
W_fc1.load(npzFile['W_fc1'], session=sess)
b_fc1.load(npzFile['b_fc1'], session=sess)
W_fc2.load(npzFile['W_fc2'], session=sess)
b_fc2.load(npzFile['b_fc2'], session=sess)
'''

# In[ ]:


## confusion matrix
y_predict = sess.run(tf.argmax(y,1), feed_dict={x: val_images,tf_keep_prob: 1.0});
y_target = sess.run(tf.argmax(val_labels,1));
print('confusion matrix:')
print(sess.run(tf.contrib.metrics.confusion_matrix(predictions = y_predict, labels = y_target)))

# In[ ]:


## final loss, accuracy 

val_loss_final = sess.run(cross_entropy, feed_dict={x: val_images,y_: val_labels, tf_keep_prob: 1.0});        
val_acc_final = accuracy.eval(feed_dict={x: val_images, y_: val_labels, tf_keep_prob: 1.0})
print('final: val_loss = %.4f, val_acc = %.4f'%(val_loss_final,val_acc_final))

plt.figure(figsize=(10, 5));
plt.subplot(1,2,1);
plt.plot(np.arange(0,len(train_acc))*log_step*mb_size/num_examples, train_acc,'-b', label='Training')
plt.plot(np.arange(0,len(val_acc))*log_step*mb_size/num_examples, val_acc,'-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 1.1, ymin = 0.0)
plt.ylabel('accuracy')
plt.xlabel('epoch');

plt.subplot(1,2,2)
plt.plot(np.arange(0,len(train_loss))*log_step*mb_size/num_examples, train_loss,'-b', label='Training')
plt.plot(np.arange(0,len(val_loss))*log_step*mb_size/num_examples, val_loss,'-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 3.0, ymin = 0.0)
plt.ylabel('loss')
plt.xlabel('epoch');

# In[ ]:


## visualize weights

W_conv1_vis = W_conv1.eval();
print('W_conv1: min = ' + str(np.min(W_conv1_vis)) + ' max = ' + str(np.max(W_conv1_vis))
      + ' mean = ' + str(np.mean(W_conv1_vis)) + ' std = ' + str(np.std(W_conv1_vis)))
W_conv1_vis = np.reshape(W_conv1_vis,(5,5,1,4,8))
W_conv1_vis = np.transpose(W_conv1_vis,(3,0,4,1,2))
W_conv1_vis = np.reshape(W_conv1_vis,(20,40,1))
plt.gca().set_xticks(np.arange(-0.5, 40, 5), minor = True);
plt.gca().set_yticks(np.arange(-0.5, 20, 5), minor = True);
plt.grid(which = 'minor', color='b', linestyle='-', linewidth=1)
plt.title('W_conv1 ' + str(W_conv1.shape))
plt.colorbar(plt.imshow(W_conv1_vis[:,:,0], cmap=cm.binary));
plt.show();

W_conv2_vis = W_conv2.eval();
print('W_conv2: min = ' + str(np.min(W_conv2_vis)) + ' max = ' + str(np.max(W_conv2_vis))
      + ' mean = ' + str(np.mean(W_conv2_vis)) + ' std = ' + str(np.std(W_conv2_vis)))
W_conv2_vis = np.reshape(W_conv2_vis,(5,5,4,8,64))
W_conv2_vis = np.transpose(W_conv2_vis,(2,0,3,1,4))
W_conv2_vis = np.reshape(W_conv2_vis,(4*5,8*5,8,8))
W_conv2_vis = np.transpose(W_conv2_vis,(2,0,3,1))
W_conv2_vis = np.reshape(W_conv2_vis,(8*4*5,8*8*5))
plt.figure(figsize=(15,10))
plt.gca().set_xticks(np.arange(-0.5, 320, 40), minor = True);
plt.gca().set_yticks(np.arange(-0.5, 160, 20), minor = True);
plt.grid(which = 'minor', color='b', linestyle='-', linewidth=1)
plt.title('W_conv2 ' + str(W_conv2.shape))
plt.colorbar(plt.imshow(W_conv2_vis[:,:], cmap=cm.binary));

#b_conv1_vis = b_conv1.eval();
#print('b_conv1 = ',b_conv1_vis)
#b_conv2_vis = b_conv2.eval();
#print('b_conv2 = ',b_conv2_vis)

# In[ ]:


## visualize activations

IMG_NO = 10;
feed_dict = {x: train_images[IMG_NO:IMG_NO+1], tf_keep_prob: 1.0}

# original image
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.title('prediction: %d'%predict.eval(feed_dict = feed_dict))
plt.imshow(train_images[IMG_NO].reshape(image_width,image_height),cmap=cm.binary);

# 1. convolution
h_conv1_vis = h_conv1.eval(feed_dict = feed_dict);
plt.subplot(2,3,2)
plt.title('h_conv1 ' + str(h_conv1_vis.shape))
h_conv1_vis = np.reshape(h_conv1_vis,(-1,28,28,4,8))
h_conv1_vis = np.transpose(h_conv1_vis,(0,3,1,4,2))
h_conv1_vis = np.reshape(h_conv1_vis,(-1,4*28,8*28))
plt.imshow(h_conv1_vis[0], cmap=cm.binary);

# 1. max pooling
h_pool1_vis = h_pool1.eval(feed_dict = feed_dict);
plt.subplot(2,3,3)
plt.title('h_pool1 ' + str(h_pool1_vis.shape))
h_pool1_vis = np.reshape(h_pool1_vis,(-1,14,14,4,8))
h_pool1_vis = np.transpose(h_pool1_vis,(0,3,1,4,2))
h_pool1_vis = np.reshape(h_pool1_vis,(-1,4*14,8*14))
plt.imshow(h_pool1_vis[0], cmap=cm.binary);

# 2. convolution
h_conv2_vis = h_conv2.eval(feed_dict = feed_dict);
plt.subplot(2,3,4)
plt.title('h_conv2 ' + str(h_conv2_vis.shape))
h_conv2_vis = np.reshape(h_conv2_vis,(-1,14,14,8,8))
h_conv2_vis = np.transpose(h_conv2_vis,(0,3,1,4,2))
h_conv2_vis = np.reshape(h_conv2_vis,(-1,8*14,8*14))
plt.imshow(h_conv2_vis[0], cmap=cm.binary);

# 2. max pooling
h_pool2_vis = h_pool2.eval(feed_dict = feed_dict);
plt.subplot(2,3,5)
plt.title('h_pool2 ' + str(h_pool2_vis.shape))
h_pool2_vis = np.reshape(h_pool2_vis,(-1,7,7,8,8))
h_pool2_vis = np.transpose(h_pool2_vis,(0,3,1,4,2))
h_pool2_vis = np.reshape(h_pool2_vis,(-1,8*7,8*7))
plt.imshow(h_pool2_vis[0], cmap=cm.binary);

# 3. FC layer
h_fc1_vis = h_fc1.eval(feed_dict = feed_dict);
plt.subplot(2,3,6)
plt.title('h_fc1 ' + str(h_fc1_vis.shape))
h_fc1_vis = np.reshape(h_fc1_vis,(-1,32,32))
plt.imshow(h_fc1_vis[0], cmap=cm.binary);
plt.show()

# 4. FC layer
h_fc2_vis = y.eval(feed_dict = feed_dict);
np.set_printoptions(precision=2)
print('h_fc2 = ', h_fc2_vis)

# ## Testing

# In[ ]:


# read test data from CSV file 
if os.path.isfile('../input/test.csv'):
    test_data = pd.read_csv('../input/test.csv') # on kaggle 
    print('test.csv loaded: test_data({0[0]},{0[1]})'.format(test_data.shape))
elif os.path.isfile('data/test.csv'):
    test_data = pd.read_csv('data/test.csv') # on local environment
    print('test.csv loaded: test_data({0[0]},{0[1]})'.format(test_data.shape))
else:
    print('Error: test.csv not found')
    
test_images = test_data.iloc[:,0:].values # (28000,784) array
test_images = test_images.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0) # convert from [0:255] => [0.0:1.0]
print('read: test_images({0[0]},{0[1]})'.format(test_images.shape));


# using mini batches is more resource efficient
predicted_labels = np.zeros(test_images.shape[0])
BATCH_SIZE = 1000;
for i in range(0,int(test_images.shape[0]/BATCH_SIZE)):
    predicted_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], tf_keep_prob: 1.0})
print('compute predicted_labels({0})'.format(len(predicted_labels)))

# save predictions
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_labels], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

print('saved: submission.csv');

# In[ ]:


# look at test images and predicted labels
plt.figure(figsize=(10,15))
for j in range(0,5):
    for i in range(0,10):
        plt.subplot(10,10,j*10+i+1)
        plt.title('%d'%predicted_labels[j*10+i])
        plt.imshow(test_images[j*10+i].reshape(28,28),cmap=cm.binary)


# In[ ]:


sess.close()
