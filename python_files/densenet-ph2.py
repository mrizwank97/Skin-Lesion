import keras
import random
import pickle
import numpy as np
import scipy.ndimage
import tensorflow as tf
import os
from PIL import Image
from keras import metrics
from random import shuffle
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.layers import  Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tiramisu_net import Tiramisu

def build_callbacks():
    checkpointer = ModelCheckpoint(filepath="../models/ph2-fcdn-bin-iou.h5", monitor='val_mean_iou', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_iou', factor=0.05, patience=4, mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_mean_iou', min_delta=1e-4, patience=10, mode='max')
    csv = keras.callbacks.CSVLogger('../logs/ph2-fcdn-bin-iou.csv', separator=',')
    callbacks = [checkpointer, reduce, early, csv]
    return callbacks

#def mean_iou(y_true, y_pred):
#    yt0 = y_true[:,:,:,0]
#    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
#    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
#    union = tf.count_nonzero(tf.add(yt0, yp0))
#    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
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

h = 256
w = 256
seed = 1

X_path = '/scratch/mraza/PH2 Dataset images/training_data/'
Y_path = '/scratch/mraza/PH2 Dataset images/training_data/'
X_val_path = '/scratch/mraza/PH2 Dataset images/validation_data/'
Y_val_path = '/scratch/mraza/PH2 Dataset images/validation_data/'
X_test_path = '/scratch/mraza/PH2 Dataset images/test_data/'
Y_test_path ='/scratch/mraza/PH2 Dataset images/test_data/'

batch_size = 4

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
    interpolation='nearest',
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    Y_path,
    target_size=(h, w),
    classes = ['masks'],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    interpolation='nearest',
    seed=seed)

train_generator = zip(image_generator, mask_generator)

image_generator = image_datagen.flow_from_directory(
    X_val_path,
    target_size=(h, w),
    classes = ['images'],
    batch_size=batch_size,
    class_mode=None,
    interpolation='nearest',
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    Y_val_path,
    target_size=(h, w),
    classes = ['masks'],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    interpolation='nearest',
    seed=seed)

val_generator=zip(image_generator, mask_generator)
image_generator = image_datagen.flow_from_directory(
    X_test_path,
    target_size=(h, w),
    classes = ['images'],
    batch_size=batch_size,
    class_mode=None,
    interpolation='nearest',
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    Y_test_path,
    target_size=(h, w),
    classes = ['masks'],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    interpolation='nearest',
    seed=seed)

test_generator=zip(image_generator, mask_generator)

model = Tiramisu(input_shape=(256,256,3),n_classes=1)
model.compile(Adam(), loss='binary_crossentropy', metrics=[mean_iou])
history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = 150//batch_size, 
                    validation_data=val_generator,
                    validation_steps=25//batch_size,
                    epochs = 100,
                    callbacks = build_callbacks(),
                    verbose=2)
loss, iou = model.evaluate_generator(test_generator,25//batch_size)
print('loss is '+str(loss))
print('miou is '+str(iou))