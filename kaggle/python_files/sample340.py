#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from livelossplot.keras import PlotLossesCallback
from keras.preprocessing.image import ImageDataGenerator

# In[ ]:


#Importing training scans of the data set as a basics to train Keras model

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, vertical_flip=True)
train_generator=train_datagen.flow_from_directory(
    '../input/chest_xray/chest_xray/train',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# In[ ]:


#Importing validation scans of the data set
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator=val_datagen.flow_from_directory(
    '../input/chest_xray/chest_xray/val',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# In[ ]:


#Plot function to visualise NORMAL and PNEUMONIA SCANS
def plot_images(item_dir, top=25):
  all_item_dirs = os.listdir(item_dir)
  item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:top]

  plt.figure(figsize = (12,12))
  for idx, img_path in enumerate(item_files):
    plt.subplot(5,5, idx+1)
    img = mpimg.imread(img_path)
    plt.imshow(img)
  plt.tight_layout()

# In[ ]:


#NORMAL training data scans plot
plot_images("../input/chest_xray/chest_xray/train/NORMAL/")

# In[ ]:


#PNEUMONIA training data scans plot
plot_images("../input/chest_xray/chest_xray/train/PNEUMONIA/")

# In[ ]:


#Keras Classification Model Definition

def myModel(input_shape):
   model = Sequential ([
    Conv2D(128, (3,3), input_shape=input_shape),
    MaxPool2D((2,2)),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    
    #Conv2D(32, (3,3), activation='relu'),   
    #MaxPool2D((2,2)),
    
    Flatten(),

    Dense(512, activation='relu'),
       
    Dropout(0.5),
       
    Dense(128, activation='relu'),
       
    Dropout(0.5),
       
    Dense(2, activation='sigmoid')   
]) 
   return model

input_shape = (224,224,3)

model = myModel(input_shape)

#Model Compilation
model.compile( loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']
)

model.summary()

# In[ ]:


#Model training with live parameters visualisation
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=25,
      validation_data=validation_generator,
      validation_steps=len(validation_generator),
      verbose=1,
      callbacks=[PlotLossesCallback()]
)

# In[ ]:


#Definition of the outputs
print(train_generator.class_indices)

# In[ ]:


#Test Prediction on random scan
original = load_img('../input/chest_xray/chest_xray/test/PNEUMONIA/person102_bacteria_487.jpeg', target_size=(224, 224))
plt.imshow(original)
plt.show()
 
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)

processed_image = preprocess_input(image_batch.copy())
predictions = model.predict(processed_image)

result = np.argmax(predictions)

if result == 0:
    print("NORMAL")
elif result == 1:
    print("PNEUMONIA")

# In[ ]:


#Model Export
model.save('pneumoniamodel.h5')
