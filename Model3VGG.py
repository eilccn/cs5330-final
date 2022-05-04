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
#print(train_df.head())

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

#y_batch = np.expand_dims(y_batch,axis=-1)
# print(x_batch.shape)
# print(y_batch.shape)
# print(type(x_batch))
# print(type(y_batch))
# print(type(x_batch[0]))
# print(type(y_batch[0][0]))
# print(y_batch)

#
# for i in range (0,5):
#      image = x_batch[i]
#      cv2.imshow("s",image)
#      cv2.waitKey(50)
#     print(y_batch[i][0].shape)
#     plt.imshow(image.astype(np.uint8))
#     plt.show()


model = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights=None,

    input_shape=(256,256,3),
    pooling='avg',



)



for layer in model.layers[:-3]:
    layer.trainable = False

x = model.output
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)

final_model = Model(inputs= model.input, outputs=y)

opt = Adam(lr=0.0001, name = "Adam")
#Compiling the final model
final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy'},

                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy']}) # default: top-5

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

final_model.fit(custom_generator(train_generator),steps_per_epoch=20, epochs=3,
                validation_data = custom_generator(val_generator), validation_steps=32)

final_model.save('model3')


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