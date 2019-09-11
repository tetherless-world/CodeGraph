#!/usr/bin/env python
# coding: utf-8

# ## CNN Architectures : VGG, Resnet, InceptionNet, XceptionNet 
# 
# ### UseCases : Image Feature Extraction + Transfer Learning
# 
# 
# <br>
# 
# A Gold mine dataset for comuter vision is the ImageNet dataset. It consists of about 14 M hand-labelled annotated images which contains over 22,000 day-to-day categories. Every year ImageNet competition is hosted in which the smaller version of this dataset (with 1000 categories) is used with an aim to accurately classify the images. Many winning solutions of the ImageNet Challenge have used state of the art convolutional neural network architectures to beat the best possible accuracy thresholds. In this kernel, I have discussed these popular architectures such as VGG16, 19, ResNet, AlexNet etc. In the end, I have explained how to generate image features using pretrained models and use them in machine learning models. 
# 
# ## Contents 
# 
# <br>
# 
# From the high level perspective, I have discussed three main components 
# 
# <ul>
#     <li>1. CNN Architectures   </li>
# <ul>
#     <li>1. 1 VGG16</li>
#     <li>1.2 VGG19 </li>
#     <li>1.3 InceptionNet</li>
#     <li>1.4 Resnet </li>
#     <li>1.5 XceptionNet</li>
# </ul>
#     <li>2. Image Feature Extraction  </li>
#     <li>3. Transfer Learning  </li>
# </ul>
# 
# 
# ## 1. CNN Architectures
# ## 1.1 &nbsp;&nbsp; VGG16 
# 
# VGG16 was publised in 2014 and is one of the simplest (among the other cnn architectures used in Imagenet competition). It's Key Characteristics are:   
# 
# 1. This network contains total 16 layers in which weights and bias parameters are learnt.    
# 2. A total of 13 convolutional layers are stacked one after the other and 3 dense layers for classification.     
# 3. The number of filters in the convolution layers follow an increasing pattern (similar to decoder architecture of autoencoder).     
# 4. The informative features are obtained by max pooling layers applied at different steps in the architecture.    
# 5. The dense layers comprises of 4096, 4096, and 1000 nodes each.   
# 6. The cons of this architecture are that it is slow to train and produces the model with very large size.   
# 
# The VGG16 architecture is given below: 
# 
# ![](https://tech.showmax.com/2017/10/convnet-architectures/image_0-8fa3b810.png)
# 
# ## Implementation : VGG16
# Let's see how we can create this architecture using python's keras library. The following code block shows the implementation of VGG16 in keras. 
# 

# In[ ]:


from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model

_input = Input((224,224,1)) 

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

flat   = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(1000, activation="softmax")(dense2)

vgg16_model  = Model(inputs=_input, outputs=output)

# ## PreTrained Model : VGG16
# 
# 
# 
# Keras library also provides the pre-trained model in which one can load the saved model weights, and use them for different purposes : transfer learning, image feature extraction, and object detection. We can load the model architecture given in the library, and then add all the weights to the respective layers. 
# 
# Before using the pretrained models, lets write a few functions which will be used to make some predictions. First, load some images and preprocess them. 

# In[ ]:


from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import pandas as pd 
import numpy as np 
import os 

img1 = "../input/dogs-vs-cats-redux-kernels-edition/train/cat.11679.jpg"
img2 = "../input/dogs-vs-cats-redux-kernels-edition/train/dog.2811.jpg"
img3 = "../input/flowers-recognition/flowers/flowers/sunflower/7791014076_07a897cb85_n.jpg"
img4 = "../input/fruits/fruits-360_dataset/fruits-360/Training/Banana/254_100.jpg"
imgs = [img1, img2, img3, img4]

def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img 

def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    for i in range(4):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()
    
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i,img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds  = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()

# Now, we can perform following steps : 
# 1. import VGG16 architecture from keras.applications  
# 2. Add the saved weights to the architecture 
# 3. Use model to perform predictions 

# In[ ]:


from keras.applications.vgg16 import VGG16
vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg16_model = VGG16(weights=vgg16_weights)
_get_predictions(vgg16_model)

# ## 1.2 &nbsp;&nbsp; VGG19 
# 
# VGG19 is a similar model architecure as VGG16 with three additional convolutional layers, it consists of a total of 16 Convolution layers and 3 dense layers.  Following is the architecture of VGG19 model. In VGG networks, the use of 3 x 3 convolutions with stride 1 gives an effective receptive filed equivalent to 7 * 7. This means there are fewer parameters to train. 
# 
# ![](https://cdn-images-1.medium.com/max/1600/1*cufAO77aeSWdShs3ba5ndg.jpeg)
# 
# 

# In[ ]:


from keras.applications.vgg19 import VGG19
vgg19_weights = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
vgg19_model = VGG19(weights=vgg19_weights)
_get_predictions(vgg19_model)

# ## &nbsp;&nbsp; 1.3 InceptionNets
# 
# Also known as GoogleNet consists of total 22 layers and was the winning model of 2014 image net challenge. 
# 
# - Inception modules are the fundamental block of InceptionNets. The key idea of inception module is to design good local network topology (network within a network)  
# - These modules or blocks acts as the multi-level feature extractor in which convolutions of different sizes are obtained to create a diversified feature map
# - The inception modules also consists of 1 x 1 convolution blocks whose role is to perform dimentionaltiy reduction.  
# - By performing the 1x1 convolution, the inception block preserves the spatial dimentions but reduces the depth. So the overall network's dimentions are not increased exponentially.  
# - Apart from the regular output layer, this network also consists of two auxillary classification outputs which are used to inject gradients at lower layers.  
# 
# <br><br>
# 
# The inception module is shown in the following figure:  
# 
# ![](https://hackathonprojects.files.wordpress.com/2016/09/inception_implement.png?w=649&h=337)
# 
# The complete architecture is shown below: 
# 
# ![](https://cdn-images-1.medium.com/max/2000/1*uXfC5fcbDsL0TJG4T8PsVw.png)
# 
# <br>
# 
# ### Pre-Trained Model : InceptionV3

# In[ ]:


from keras.applications.inception_v3 import InceptionV3
inception_weights = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
inception_model = InceptionV3(weights=inception_weights)
_get_predictions(inception_model)

# ## 1.4 Resnets
# 
# Original Paper : https://arxiv.org/pdf/1512.03385.pdf
# 
# All the previous models used deep neural networks in which they stacked many convolution layers one after the other. It was learnt that deeper networks are performing better. However, it turned out that this is not really true. Following are the problems with deeper networks: 
# 
# - Network becomes difficult to optimize  
# - Vanishing / Exploding Gradeints  
# - Degradation Problem ( accuracy first saturates and then degrades )  
# 
# ### Skip Connections   
# 
# So to address these problems, authors of the resnet architecture came up with the idea of skip connections with the hypothesis that the deeper layers should be able to learn something as equal as shallower layers. A possible solution is copying the activations from shallower layers and setting additional layers to identity mapping.   These connections are enabled by skip connections which are shown in the following figure. 
# 
# ![](https://cdn-images-1.medium.com/max/987/1*pUyst_ciesOz_LUg0HocYg.png)
# 
# So the role of these connections is to perform identity function over the activation of shallower layer, which in-turn produces the same activation. This output is then added with the activation of the next layer. To enable these connections or essentially enable this addition operation, one need to ensure the same dimentions of convolutions through out the network, that's why resnets have same 3 by 3 convolutions throughout. 
# 
# ### Key Advantage 
# 
# By using residual blocks in the network, one can construct networks of any depth with the hypothesis that new layers are actually helping to learn new underlying patterns in the input data. The authors of the paper were able to create the deep neural network architecture with 152 layers. The variants of Resnets such as resnet34, resnet50, resnet101 have produced the solutions with very high accuracy in Imagenet competitions. 
# 
# ### Why it works ? 
# 
# Lets discuss why residual networks are successful and enables the addition of more and more layers without the key problems ie. without hurting the network performance. 
# 
# Consider a plain neural network (A) without residual network as shown. So in the network (A) the input X is passed to this Neural Network (NN) to give the activation A1. 
# 
#   <br>
# 
# ![](https://i.imgur.com/9j8bKaY.png)
# 
# Now, consider a more deeper network (B) in which a residual block (with 2 extra layers and a skip connection) is added in the previous network. So now, the activation A1 is being passed to Residual Block which in turns gives new activation A3. 
# 
# if there was no skip connection, then A3 was: 
# 
# >  A3 = relu ( W2 . A2 + b2)              ..... (without skip connection)
# 
# where W2 and b2 are weights and bias associated with layer L2. But, with skip connection another term A1 will be passed to L2. So the equation of A3 will be modified as: 
# 
# > A3 = relu ( W2 . A2 + b2 + A1) 
# 
# If we use L2 regularization or the weight decay methods, they will force W2 and b2 to become close to zero. In the worst case, if these become zero, then 
# 
# > A3 = relu (A1)   
# 
# because relu will output 0 for negative, A1 for positive and we know that A1 is previous activation from relu which is positive. 
# 
# > A3 = A1 
# 
# This means that Identitiy function is easy for residual blocks to learn. By addition of residual blocks, model complexity was not increased. As this is only copying the previous activation to the next layers. However this is only the worst case situation, but the it may turn out that these additional layers learns something useful. In that case, the network performance will improve. 
# 
# Hence, adding the residual blocks / skip connections does not hurt the network performance but infact increases the chances that new layers will learn something useful. 
# 
# Let's look at the usage using pre-trained resnet 50 model. 

# In[ ]:


from keras.applications.resnet50 import ResNet50
resnet_weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
resnet_model = ResNet50(weights=resnet_weights)
_get_predictions(resnet_model)

# ## 1.5 Xception Nets
# 
# Xception is an extension of the Inception architecture which replaces the standard Inception modules with depthwise separable convolutions.

# In[ ]:


from keras.applications.xception import Xception
xception_weights = '../input/xception/xception_weights_tf_dim_ordering_tf_kernels.h5'
xception_model = Xception(weights=xception_weights)

# #### Comparison of different architectures: 
# 
# The following image describes the relative comparison of these architectures in terms of performance as size. 
# <br><br>
# 
# ![](http://www.houseofbots.com/images/news/573/cover.png)
# 
# <br><br>
# 
# ## 2. Image Feature Extraction using PreTrained Models 
# 
# Lets look at how one can use pre-trained models for feature extraction, The extracted features can be used for Machine Learning purposes. 
# 
# First step is to load the weights of the pre-trained model in the model architecture. Notice, that an additional argument is passed include_top = False, which states that we do not want to add the last layer of this architecture. 

# In[ ]:


resnet50 = ResNet50(weights='imagenet', include_top=False)

# As the next step, we will pass an image to this model and identify the features. 

# In[ ]:


def _get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    resnet_features = resnet50.predict(img_data)
    return resnet_features

img_path = "../input/dogs-vs-cats-redux-kernels-edition/train/dog.2811.jpg"
resnet_features = _get_features(img_path)

# now the extracted features are stored in the variable resnet_features. One can flatten them or sequee them in order to use them in ML models.  Flatten will produce a long vector of feature elements. Squeeze will produce a 3D matrix of the features

# In[ ]:


features_representation_1 = resnet_features.flatten()
features_representation_2 = resnet_features.squeeze()

print ("Shape 1: ", features_representation_1.shape)
print ("Shape 2: ", features_representation_2.shape)

# ## 3. Transfer Learning Example 
# 
# <br>
# 
# Lets look at the implemetation of transfer learning using pre-trained model features. First, we 'll create a dataset containing two classes of images : bananas and strawberrys. Also add a test dataset contianing images from both classes.
# 
# ### 3.1 Dataset Preparation

# In[ ]:


basepath = "../input/fruits/fruits-360_dataset/fruits-360/Training/"
class1 = os.listdir(basepath + "Banana/")
class2 = os.listdir(basepath + "Strawberry/")

data = {'banana': class1[:10], 
        'strawberry': class2[:10], 
        'test': [class1[11], class2[11]]}

# Transfer learning can be implemented in two steps: 
# 
# Step 1 : Image Feature Exraction  
# Step 2 : Training a Classifier  
# 
# ### Step 1 : Feature Extraction using pre-trained models (resnet50)
# 
# Iterate in the images, call the same function used in point 2 for image feature extraction, we will use the flatten representation of these features 

# In[ ]:


features = {"banana" : [], "strawberry" : [], "test" : []}
testimgs = []
for label, val in data.items():
    for k, each in enumerate(val):        
        if label == "test" and k == 0:
            img_path = basepath + "/Banana/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 1:
            img_path = basepath + "/Strawberry/" + each
            testimgs.append(img_path)
        else: 
            img_path = basepath + label.title() + "/" + each
        feats = _get_features(img_path)
        features[label].append(feats.flatten())        

# Next, Convert the features from dictionary format to pandas dataframe. A long dataframe will be created. I will be applying variance filter later on this dataframe to reduce the dimentionality. Other ideas to avoid this step : perform PCA / SVD to obtain the dense features. 

# In[ ]:


dataset = pd.DataFrame()
for label, feats in features.items():
    temp_df = pd.DataFrame(feats)
    temp_df['label'] = label
    dataset = dataset.append(temp_df, ignore_index=True)
dataset.head()

# Prepare X (predictors) and y (target) from the dataset

# In[ ]:


y = dataset[dataset.label != 'test'].label
X = dataset[dataset.label != 'test'].drop('label', axis=1)

# ### Step 2: Write a classifier to predict two classes
# 
# we will write a simple neural network (multi layer perceptron classifier) using sklearn for training purposes. 

# In[ ]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

model = MLPClassifier(hidden_layer_sizes=(100, 10))
pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
pipeline.fit(X, y)

print ("Model Trained on pre-trained features")

# Let's predict the output on new images and check the outcome.

# In[ ]:


preds = pipeline.predict(features['test'])

f, ax = plt.subplots(1, 2)
for i in range(2):
    ax[i].imshow(Image.open(testimgs[i]).resize((200, 200), Image.ANTIALIAS))
    ax[i].text(10, 180, 'Predicted: %s' % preds[i], color='k', backgroundcolor='red', alpha=0.8)
plt.show()

# So a simple neural network with only 20 rows of training data is able to correctly classify the two images on test set. 
# 
# ### EndNotes 
# Thanks for viewing this kernel, If you liked it, please upvote. 
