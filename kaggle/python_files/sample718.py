#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install opencv-python

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# # An implimentation of Facial landmark Detection using the trained data from iBUG's 68 landmark model
# [Code Curtesy of Adrian Rosebrock's pyimage blog](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
# This model identifies 68 landmarks around the face; 17 defining the shape of the jaw, 10 for the brows (5 each), 4 along the bridge of the nose, 5 along the tip and base of the nose, 12 for eye shape (6 each), 12 for the outer lip outline, 8 for the inside lip outline
# ![](https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg)
# The results of the model are 68 (x,y) points for each given face, 136 data points overall

# In[ ]:


import cv2,matplotlib.pyplot as plt,dlib,imutils
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../input/dlib-68/shape_predictor_68_face_landmarks.dat")

# In[ ]:


image=plt.imread("../input/recognizing-faces-in-the-wild/train/F0002/MID1/P00009_face3.jpg")
# image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for rect in rects:
    pred=predictor(gray,rect)
    fig, ax1 = plt.subplots()

    ax1.imshow(image)
    ax1.scatter(face_utils.shape_to_np(pred)[:,0],face_utils.shape_to_np(pred)[:,1])
    
# del predictor

# # Referential Association of relationships
# > ~~An assumption is made that for triplett loss, simply using the un-related family members is good enough to use as negative cases, if this shows to be a bad assumption, changes can be made~~
# 
# > Revision: the model had troubles coping with the trasitive non-equivelance (Person A related to B, B related to C, but C not related to A) as the only things to train on;added 3 random people to each person as unrelated persons and the model now trains nicely

# In[ ]:


import random, itertools,glob
class Person:
    def __init__(self,name,Family):
        self.name=name
        self.family=Family
        self.related=set()
        self.unrelated=set()

# In[ ]:


relationlist=open("../input/recognizing-faces-in-the-wild/train_relationships.csv").read().split("\n")[1:-1]
Families={k.split("/")[0]:{} for k in relationlist}
for each in relationlist:
    p1=each.split(",")[0].split("/")[1]
    p2=each.split("/")[2]
    Fam=each.split("/")[0]
    Families[Fam].update({p1:Person(p1,Fam),p2:Person(p2,Fam)})
Families
for Fam in Families:
    Family=Families[Fam]
    for Pers in Family:
        Families[Fam][Pers].unrelated.update([k for k in set(Family.values()) if k.name!=Pers])
for relation in relationlist:
    A,B=[Families[A.split("/")[0]][A.split("/")[1]] for A in relation.split(",")]
    a,b=[r.split("/") for r in relation.split(",")]
    Families[a[0]][a[1]].unrelated=A.unrelated - set([B])
    Families[a[0]][a[1]].related=A.related   | set([B])
    Families[b[0]][b[1]].unrelated=B.unrelated - set([A])
    Families[b[0]][b[1]].related=B.related   | set([A])
    
for F in Families:
    for P in Families[F]:
        if len(Families[F][P].unrelated)==0:
            # For those that are fully related to those in the family, randomly choose 3 other people to be unrelated to
            Families[F][P].unrelated= Families[F][P].unrelated | set([k for k in
                                                                           [random.choice(list(random.choice(list(Families.values())).values())) for s in range(len(Families[F][P].related)+3)]
                                                                          if k not in Families[F][P].related])
        if len(Families[F][P].related)==0:
            #ensure there are not any marooned individuals that are not related to anyone
            print("Related",F,P)
        
        Families[F][P].unrelated= Families[F][P].unrelated | set([random.choice(list(random.choice(list(Families.values())).values())) for s in range(3)])
# Families

# In[ ]:


train_data_parts=[]
[[train_data_parts.append(Families[F][P]) for P in Families[F]] for F in Families]
del Families
train_data_parts[:10]

# In[ ]:


def metaglob(lis):
    ret=[]
    [ret.extend(glob.glob("../input/recognizing-faces-in-the-wild/train/"+A.family+"/"+A.name+"/*.jpg")) for A in lis]
    return ret

pairs=[]
[pairs.extend(itertools.product(*[metaglob([A]),
                                  metaglob(A.related),
                                  metaglob(A.unrelated)])) for A in train_data_parts]
del train_data_parts
print("Done")

# In[ ]:


len(pairs)

# *OOFF* 65-75 Million photo pairs, probably excessive, lets go with some subset for now

# In[ ]:


from keras.utils.generic_utils import Progbar

def SixtyEight(image,k):
    k.add(1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        pred=predictor(gray,rects[0])
        re=face_utils.shape_to_np(pred)
        re=(re-re.min(0))/(re.max(0)-re.min(0))
        return re
    return None


# In[ ]:


class callable_dict:
    def __init__(self,fun):
        self.dict=dict()
        self.function=fun
    def __getitem__(self, key):
        if key in self.dict.keys():
            return self.dict[key]
        else:
            self.dict[key]=self.function(plt.imread(key),Progbar(target=1, verbose=0))
            return self.dict[key]

# In[ ]:


finalpair=pairs[::300]

train=[[] for i in range(3)]
s=Progbar(target=len(finalpair))

myDict=callable_dict(SixtyEight)

for p in finalpair:
    s.add(1)
    one,two,three=[myDict[x] for x in p]
    if False not in [type(k)==np.ndarray for k in [one,two,three]]:
        train[0].append(one)
        train[1].append(two)
        train[2].append(three)

train=[np.array(k) for k in train]

# In[ ]:


train[1].shape

# # Triplet loss biometric loss function
# [Curtousey of Towards DataScience](https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24)

# >![Demonstration](https://cdn-images-1.medium.com/max/800/0*_WNBFcRVEOz6QM7R.)

# In[ ]:


import tensorflow as tf
def triplet_loss(y_true, y_pred, alpha = 400,N=5):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
#     print((y_pred[0]))
    N=y_pred.shape[1]//3
    anchor = y_pred[:,0:N]
    positive = y_pred[:,N:N*2]
    negative = y_pred[:,N*2:N*3]

    # distance between the anchor and the positive
    pos_dist = K.sqrt(K.sum(K.square(anchor-positive),axis=1)+.01)

    # distance between the anchor and the negative
    neg_dist = K.sqrt(K.sum(K.square(anchor-negative),axis=1)+.01)

    # compute loss
    basic_loss = (pos_dist-neg_dist+alpha)
    loss = K.maximum(basic_loss,0.0)
 
    return loss

def Neg_Dist(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
#     print((y_pred[0]))
    N=y_pred.shape[1]//3
    anchor = y_pred[:,0:N]
    negative = y_pred[:,N*2:N*3]



    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

 
    return neg_dist


# In[ ]:


from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Conv2D
from keras.optimizers import Adagrad, Adam
from keras.metrics import K

# In[ ]:


def create_mod(inpu,outpu):
    model= Sequential()
    model.add(Dense(256, input_shape=(68,2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
#     model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
#     model.add(Dropout(rate=0.1))

    model.add(Dense(outpu))
    model.add(Activation("relu"))
    return model

with tf.device('/device:GPU:1'):
    anchor_in,pos_in,neg_in = Input(shape=(68,2)),Input(shape=(68,2)),Input(shape=(68,2))

    mod=create_mod(224,68)

    anchor_out=mod(anchor_in)
    pos_out=mod(pos_in)
    neg_out=mod(neg_in)

    merged= concatenate([anchor_out,pos_out,neg_out], axis=-1)

    model=Model(inputs=[anchor_in,pos_in,neg_in],outputs=merged)

    model.compile(loss=triplet_loss,optimizer=Adam())

model.fit(train,np.zeros(train[0].shape[0]),batch_size=100,epochs=10)

# In[ ]:


pre=model.predict([train[0],train[1],train[2]])

# In[ ]:


    N=68
    anchor = pre[:,0:N]
    positive = pre[:,N:N*2]
    negative = pre[:,N*2:N*3]

# In[ ]:


np.sqrt(np.square(anchor-positive).sum(1)).mean(),np.sqrt(np.square(anchor-negative).sum(1)).mean()

# In[ ]:


p=pre.reshape((train[1].shape[0],3,68))


# In[ ]:


np.linalg.norm(p[400,0]-p[0,0],2)


# In[ ]:


p[0,0].shape

# In[ ]:


holdoutpairs=pairs[150::300]

testset=[[] for i in range(3)]
s=Progbar(target=len(finalpair))


for p in finalpair:
    s.add(1)
    one,two,three=[myDict[x] for x in p]
    if False not in [type(k)==np.ndarray for k in [one,two,three]]:
        testset[0].append(one)
        testset[1].append(two)
        testset[2].append(three)

holdout=[np.array(k) for k in testset]
del testset,holdoutpairs

HO=model.predict(holdout).reshape((holdout[1].shape[0],3,68))

# In[ ]:


test=list(pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")["img_pair"])
tests=set()
for k in test:
    tests=tests | set(k.split("-")) 
sorted(tests)
k=Progbar(len(tests))
comp={fil:SixtyEight(plt.imread("../input/recognizing-faces-in-the-wild/test/"+fil),k) for fil in tests}
# comp

# In[ ]:


test1=[]
test2=[]
keys=[]
for face in test:
    one=comp[face.split("-")[0]]
    two=comp[face.split("-")[1]]
    if type(one) == np.ndarray and type(two) == np.ndarray:
        test1.append(one)
        test2.append(two)
        keys.append(face)
test1=np.array(test1)
test2=np.array(test2)

# In[ ]:


te=model.predict([test1,test1,test2])

# In[ ]:


test1.shape
o=te.reshape((test1.shape[0],3,68))
np.array([[np.linalg.norm(k[0]-k[1],2),np.linalg.norm(k[0]-k[2],2)] for k in o]).mean(0)

# In[ ]:


kk=np.array([[np.linalg.norm(k[0]-k[1],2),np.linalg.norm(k[0]-k[2],2)] for k in p])
plt.hist(kk,bins=20)
plt.legend(["Related","Unrelated"])
plt.title("Training Data Distributions")

# In[ ]:


kk=np.array([[np.linalg.norm(k[0]-k[1],2),np.linalg.norm(k[0]-k[2],2)] for k in HO])
plt.hist(kk,bins=20)
plt.legend(["Related","Unrelated"])
plt.title("Holdout Data Distributions")

# In[ ]:


kk=np.array([[np.linalg.norm(k[0]-k[1],2),np.linalg.norm(k[0]-k[2],2)] for k in o])[:,1]
plt.hist(kk,bins=20,color="k")

# In[ ]:


ppd=pd.DataFrame({"img_pair":keys,"is_related":1-(kk-kk.min())/(kk.max()-kk.min())})
ppd=ppd.append(pd.DataFrame({"img_pair":list(set(test)-set(keys)),"is_related":[1-(kk.mean()-kk.min())/(kk.max()-kk.min()) for i in range(len(list(set(test)-set(keys))))]}))
ppd.to_csv("submission.csv",index=False,header=True)
