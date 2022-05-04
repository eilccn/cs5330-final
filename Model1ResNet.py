from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#creates and stores the model

import pandas as pd
import cv2
import numpy as np
#from utils import prepare_dataframe
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K



train_df = pd.read_csv("split-data/train_new.csv")
val_df = pd.read_csv("split-data/val_new.csv")
test_df = pd.read_csv("split-data/test_new.csv")


traingen=ImageDataGenerator(rescale= 1./255 , height_shift_range=0.2, rotation_range = 10,
                           horizontal_flip = True,zoom_range= [0.7,1.3])
train_generator=traingen.flow_from_dataframe(
dataframe=train_df,
directory="dataset",
x_col="img_path",
y_col= "category",
batch_size=128,
shuffle= True,
target_size = (256,256))


valgen = ImageDataGenerator(rescale= 1./255)
val_generator=valgen.flow_from_dataframe(
dataframe=val_df,
directory="dataset",
x_col="img_path",
y_col= "category",
batch_size=128,
shuffle= True,
target_size = (256,256))


x_batch, y_batch = next(train_generator)




model_resnet = ResNet50(weights= None, include_top=False, pooling='avg', input_shape = (256,256,3))

for layer in model_resnet.layers[:-3]:
    layer.trainable = False


x = model_resnet.output
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)



final_model = Model(inputs= model_resnet.input, outputs=[y])

opt = Adam(lr=0.05, name = "Adam")

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy' },
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('./models/model.h5')
def custom_generator(iterator):
    while True:
        batch_x, batch_y = next(iterator)
        batch_y = np.expand_dims(y_batch,axis=-1)
        yield (batch_x,batch_y)
#Fitting the model on the training sata
final_model.fit(custom_generator(train_generator),steps_per_epoch=20, epochs=3,
                validation_data = custom_generator(val_generator), validation_steps=32)

final_model.save('model1')


testgen = ImageDataGenerator()

test_generator=testgen.flow_from_dataframe(
dataframe=test_df,
directory="./dataset",
x_col="img_path",
y_col= "category",
batch_size=128,
shuffle= True,
target_size = (256,256))

results = final_model.evaluate(custom_generator(test_generator), steps=10)

print(results)