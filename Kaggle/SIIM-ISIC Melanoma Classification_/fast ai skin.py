# -*- coding: utf-8 -*-
# The above encoding declaration is required and the file must be saved as UTF-8

import pandas as pd
import os
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sp
import seaborn as sns
import glob

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.preprocessing import image
from keras import backend as K 
from tensorflow import keras
  
img_width, img_height = 224, 224
  
train_data_dir = "/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/train2"
#validation_data_dir = "/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/test"
#nb_train_samples = 32542+584
#nb_validation_samples = 0

nb_train_samples = 928+584
#nb_validation_samples = 12
epochs = 5
batch_size = 16 #32
  

if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 

"""
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 
#model.add(Dense(1, init='uniform', activation='linear'))
  
model.compile(loss ='binary_crossentropy', 
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 
                 


train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2, 
            horizontal_flip = True) 

"""

"""
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = epochs)

"""
"""
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='binary') 


model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, )


print(model.summary())
"""


"""
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='binary') 

"""
""" 
validation_generator = test_datagen.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='binary') 
"""

""" 
model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, 
    validation_steps = nb_validation_samples // batch_size) 
"""

"""
#Save model
model.save('model2')
"""


#Load model 

model = keras.models.load_model('model2')
print(model.summary())


# predicting images
#img = image.load_img('/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/train/benign/ISIC_0076262.jpg', target_size=(img_width, img_height))
img = image.load_img('/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/train/benign/ISIC_0324600.jpg', target_size=(img_width, img_height))
#img = image.load_img('/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/train/malignant/ISIC_0369831.jpg', target_size=(img_width, img_height))
#img = image.load_img('/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/train/malignant/ISIC_5140952.jpg', target_size=(img_width, img_height))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255
images = np.vstack([x])


classes = model.predict_classes(images, batch_size=32)
print (classes)

predictions = model.predict(images)
print(predictions)

# make a prediction
ynew = model.predict_proba(images)
# show the inputs and predicted outputs
print(ynew[0][0])

predictions = model.predict(images)
score = predictions[0]
good=score*100
print(good)

print(
    "This image is "+str(good)+" percent malignant."
 
)

sample_submission=pd.read_csv('/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/sample_submission.csv')
path1='/media/linux/Disque Dur/DOWNLOADS/siim-isic-melanoma-classification/Data done/test'

listing = os.listdir(path1)  
name=[]
for file in listing:
    img = image.load_img(path1+"/"+file,target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    images = np.vstack([x])
    predictions = model.predict(images)
    score = predictions[0]
    score=float(score)
    score=round(score,3)
    good=score*100
    item=str(file)
    item=item.replace(".jpg","")
    name.append({'image_name':item,'target':score})
    print(item)
    print(
        "This image is "+str(good)+" percent malignant."
    
    )

df = pd.DataFrame(name, columns = ['image_name', 'target'])
df.to_csv("final.csv", index=False)



"""
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv("final.csv", index=False)
"""