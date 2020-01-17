import warnings
warnings.filterwarnings('ignore')
import os
import keras
import random
import pickle
import numpy as np
import scipy.ndimage
from PIL import Image
import tensorflow as tf
from keras import metrics
from random import shuffle
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.layers import  Dropout
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import concatenate


def build_callbacks():
    checkpointer = ModelCheckpoint(filepath="../models/isic-unet-bin-iou.h5", monitor='val_mean_iou', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_iou', factor=0.05, patience=5, mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_mean_iou', min_delta=1e-4, patience=16, mode='max')
    csv = keras.callbacks.CSVLogger('../logs/isic-unet-bin-iou.csv', separator=',')
    callbacks = [checkpointer, reduce, early, csv]
    return callbacks

#def mean_iou(y_true, y_pred):
#    yt0 = y_true[:,:,:,0]
#    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
#    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
#    union = tf.count_nonzero(tf.add(yt0, yp0))
#    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
#    return iou

#def mean_iou(y_true, y_pred, smooth=.001):
#    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#    return iou

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def unet(input_layer, features):
    #downsampling 1
    conv1 = Conv2D(features, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(features, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)    #drop 25% to 0 to avoid overfit
 
    #downsampling 2
    conv2 = Conv2D(features * 2, (3, 3), activation='relu', padding='same') (pool1)
    conv2 = Conv2D(features * 2, (3, 3), activation='relu', padding='same') (conv2)
    pool2 = MaxPooling2D((2, 2)) (conv2)
    pool2 = Dropout(0.3)(pool2)
 
    #downsampling 3
    conv3 = Conv2D(features * 2**2, (3, 3), activation='relu', padding='same') (pool2)
    conv3 = Conv2D(features * 2**2, (3, 3), activation='relu', padding='same') (conv3)
    pool3 = MaxPooling2D((2, 2)) (conv3)   
    pool3 = Dropout(0.4)(pool3)
 
    #downsampling 4
    conv4 = Conv2D(features * 2**3, (3, 3), activation='relu', padding='same') (pool3)
    conv4 = Conv2D(features * 2**3, (3, 3), activation='relu', padding='same') (conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (conv4)
    pool4 = Dropout(0.5)(pool4)
 
    #middle bridge 5
    conv5 = Conv2D(features * 2**4, (3, 3), activation='relu', padding='same') (pool4)
    conv5 = Conv2D(features * 2**4, (3, 3), activation='relu', padding='same') (conv5)
 
    #upsampling 6
    tran6 = Conv2DTranspose(features * 2**3, (2, 2), strides=(2, 2), padding='same') (conv5)
    tran6 = concatenate([tran6, conv4])   #merge tran6 and conv4 layers into 1 layer
    tran6 = Dropout(0.5)(tran6)
    conv6 = Conv2D(features * 2**3, (3, 3), activation='relu', padding='same') (tran6)
    conv6 = Conv2D(features * 2**3, (3, 3), activation='relu', padding='same') (conv6)
 
    #upsampling 7
    tran7 = Conv2DTranspose(features * 2**2, (2, 2), strides=(2, 2), padding='same') (conv6)
    tran7 = concatenate([tran7, conv3])   #merge 2 layers into 1 
    tran7 = Dropout(0.4)(tran7)
    conv7 = Conv2D(features * 2**2, (3, 3), activation='relu', padding='same') (tran7)
    conv7 = Conv2D(features * 2**2, (3, 3), activation='relu', padding='same') (conv7)
 
    #upsampling 8
    tran8 = Conv2DTranspose(features * 2, (2, 2), strides=(2, 2), padding='same') (conv7)
    tran8 = concatenate([tran8, conv2])   #merge 2 layers into 1 
    tran8 = Dropout(0.3)(tran8)
    conv8 = Conv2D(features * 2, (3, 3), activation='relu', padding='same') (tran8)
    conv8 = Conv2D(features * 2, (3, 3), activation='relu', padding='same') (conv8)
 
    #upsampling 9
    tran9 = Conv2DTranspose(features, (2, 2), strides=(2, 2), padding='same') (conv8)
    tran9 = concatenate([tran9, conv1])   #merge 2 layers into 1 
    tran9 = Dropout(0.25)(tran9)
    conv9 = Conv2D(features, (3, 3), activation='relu', padding='same') (tran9)
    conv9 = Conv2D(features, (3, 3), activation='relu', padding='same') (conv9)
 
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(conv9)

    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
 
    return model

h = 256
w = 256
seed = 1
X_path = '/scratch/mraza/Skin/training_data/'
Y_path = '/scratch/mraza/Skin/training_data/'
X_val_path = '/scratch/mraza/Skin/validation_data/'
Y_val_path = '/scratch/mraza/Skin/validation_data/'
X_test_path = '/scratch/mraza/Skin/test_data/'
Y_test_path = '/scratch/mraza/Skin/test_data/'
batch_size = 32
interpolation = 'bicubic'
x_gen_args = dict(
                    rescale=1./255,
                    rotation_range=0.2,
                    shear_range=0.3,
                    zoom_range=0.3,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                 )

y_gen_args = dict(
                    rescale=1./255,
                    rotation_range=0.2,
                    shear_range=0.3,
                    zoom_range=0.3,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                 )

image_datagen = ImageDataGenerator(**x_gen_args)
mask_datagen = ImageDataGenerator(**y_gen_args)

image_generator = image_datagen.flow_from_directory(
    X_path,
    target_size=(h, w),
    classes = ['images'],
    batch_size=batch_size,
    class_mode=None,
    interpolation=interpolation,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    Y_path,
    target_size=(h, w),
    classes = ['masks'],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    interpolation=interpolation,
    seed=seed)

train_generator = zip(image_generator, mask_generator)

image_generator = image_datagen.flow_from_directory(
    X_val_path,
    target_size=(h, w),
    classes = ['images'],
    batch_size=batch_size,
    class_mode=None,
    interpolation=interpolation,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    Y_val_path,
    target_size=(h, w),
    classes = ['masks'],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    interpolation=interpolation,
    seed=seed)

val_generator=zip(image_generator, mask_generator)

image_generator = image_datagen.flow_from_directory(
    X_test_path,
    target_size=(h, w),
    classes = ['images'],
    batch_size=batch_size,
    class_mode=None,
    interpolation=interpolation,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    Y_test_path,
    target_size=(h, w),
    classes = ['masks'],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    interpolation=interpolation,
    seed=seed)

test_generator=zip(image_generator, mask_generator)

input_layer = Input((w, h, 3))
model = unet(input_layer, 64)
history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = 1894//batch_size, 
                    validation_data=val_generator,
                    validation_steps=350//batch_size,
					epochs = 200,
                    callbacks = build_callbacks(),
					verbose=2
                    )
loss, iou  = model.evaluate_generator(test_generator, steps=350//batch_size)
print(loss)
print(iou)
#raw = Image.open('sls_test.jpg')
#raw = np.array(raw.resize((256, 256)))/255.
#raw = raw[:,:,0:3]

#predict the mask 
#pred = model.predict(np.expand_dims(raw, 0))

#mask post-processing 
#msk  = pred.squeeze()
#msk = np.stack((msk,)*3, axis=-1)
#msk[msk >= 0.5] = 1 
#msk[msk < 0.5] = 0 

#show the mask and the segmented image 
#combined = np.concatenate([raw, msk, raw* msk], axis = 1)
#plt.imshow(combined)
#plt.savefig('test_result.png')
