

# import libriay 

import pathlib
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Input, Sequential
from tensorflow.keras.optimizers import Adam

#Load database form local computer 
data_directory = pathlib.Path('./Data for test')
class_names = ['Bed','Chair','Sofa']

data_dir = './Data for test'

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(rescale = 1/255., validation_split = 0.1)

#Generate training data
train_data = data_gen.flow_from_directory(data_dir, target_size = (224, 224), subset = 'training', class_mode = 'binary')
#Generate validation data
val_data = data_gen.flow_from_directory(data_dir, target_size = (224, 224), subset = 'validation', class_mode = 'binary')

#Generate models under the appropriate parameters
model = Sequential([
    Input(shape = (224, 224, 3)),
    Conv2D(5, (3, 3), padding = 'valid',activation='relu'),
    MaxPool2D(pool_size = 2),
    Flatten(),
    Dropout(0.5),
    (Dense(101, activation='softmax'))
])

# compile the model 
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#Fit with appropriate parameters
history = model.fit(train_data,epochs= 5,steps_per_epoch = len(train_data),validation_data = val_data,validation_steps = len(val_data))

# Save the model
model.save('fulhaus-test.h5')