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
    checkpointer = ModelCheckpoint(filepath="../models/fcdn-isic-500-tl-gn.h5", monitor='val_mean_iou', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_iou', factor=0.05, patience=4, mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_mean_iou', min_delta=1e-4, patience=10, mode='max')
    csv = keras.callbacks.CSVLogger('../logs/fcdn-isic-500-tl-gn.csv', separator=',')
    callbacks = [checkpointer, reduce, early, csv]
    return callbacks

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

def dsc(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def tanimoto_loss(y_true, y_pred):  
    tanimoto = (dice_loss(y_true, y_pred) + dice_loss(1-  y_true,1 - y_pred))/2
    return tanimoto

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

h = 192
w = 256
seed = 1

X_path = '/scratch/mraza/Skin/training_data/'
Y_path = '/scratch/mraza/Skin/training_data/'
X_val_path = '/scratch/mraza/Skin/validation_data/'
Y_val_path = '/scratch/mraza/Skin/validation_data/'
X_test_path = '/scratch/mraza/Skin/test_data/'
Y_test_path = '/scratch/mraza/Skin/test_data/'
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

model = Tiramisu(input_shape=(192,256,3),n_classes=1,attention=True)
model.compile(Adam(), loss=[tanimoto_loss], metrics=[mean_iou])
#model.compile(Adam(), loss=[focal_tversky], metrics=[mean_iou])


history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = 1894//batch_size, 
                    validation_data=val_generator,
                    validation_steps=350//batch_size,
                    epochs = 500,
                    callbacks = build_callbacks(),
                    verbose=2
                    )

loss, acc = model.evaluate_generator(test_generator,350//batch_size)

print('loss is '+str(loss))
print('iou is '+str(acc))
