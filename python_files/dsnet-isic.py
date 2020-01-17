import keras
import random
import pickle
import numpy as np
import scipy.ndimage
import tensorflow as tf
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
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense, ZeroPadding2D, Input, Activation, Dropout, GlobalAveragePooling2D, BatchNormalization, concatenate, AveragePooling2D, SeparableConv2D
from group_norm import GroupNormalization
from custom_layers import Scale

def build_callbacks():
    checkpointer = ModelCheckpoint(filepath="../models/isic-dsnet-bin-iou.h5", monitor='val_mean_iou', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_iou', factor=0.05, patience=4, mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_mean_iou', min_delta=1e-4, patience=10, mode='max')
    csv = keras.callbacks.CSVLogger('../logs/isic-dsnet-bin-iou.csv', separator=',')
    callbacks = [checkpointer, reduce, early, csv]
    return callbacks

def iou_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def iou_bce(y_true, y_pred):
    iou = iou_loss(y_true,y_pred)
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return iou+bce

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

h = 192
w = 256
seed = 1

X_path = '/scratch/mraza/Skin/training_data/'
Y_path = '/scratch/mraza/Skin/training_data/'
X_val_path = '/scratch/mraza/Skin/validation_data/'
Y_val_path = '/scratch/mraza/Skin/validation_data/'
X_test_path = '/scratch/mraza/Skin/test_data/'
Y_test_path = '/scratch/mraza/Skin/test_data/'
batch_size = 16

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

def conv_layer(conv_x, filters, name):
    conv_x = BatchNormalization(axis=3, name='conv_'+name+'_bn1')(conv_x)
    conv_x = Scale(axis=3, name='conv_'+name+'_scale1')(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters*4, (1, 1),name='conv_'+name+'_1x1', kernel_initializer='he_uniform', use_bias=False)(conv_x)
    conv_x = BatchNormalization(axis=3, name='conv_'+name+'_bn3')(conv_x)
    conv_x = Scale(axis=3, name='conv_'+name+'_scale3')(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = ZeroPadding2D((1, 1), name='conv_'+name+'_zeropadding3')(conv_x)
    conv_x = Conv2D(filters, (3, 3), name='conv_'+name+'_3x3', kernel_initializer='he_uniform', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)
    return conv_x

def dense_block(block_x, filters, growth_rate, layers_in_block, name):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate, name + '_'+str(i))
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters

def dsnet(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_img = Input(shape=(192, 256, 3))
  
    dense_x = ZeroPadding2D((3, 3), name='conv1_zero_padding')(input_img)
    conv = Conv2D(filters, (7, 7), strides=(2,2), name='conv1_7x7', kernel_initializer='he_uniform', use_bias=False)(dense_x)
    dense_x = BatchNormalization(axis=3, name='conv1_bn1')(conv)
    dense_x = Scale(axis=3, name='conv1_scale')(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(dense_x)
    pool = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(dense_x)
    sfilters = filters
    
    #dense block 1
    dense_x, filters = dense_block(pool, filters, growth_rate, layers_in_block,'db1')
    
    #transition block 1
    c1filters = filters // 2
    trans_x = BatchNormalization(axis=3, name='conv_tb1_bn1')(dense_x)
    trans_x = Scale(axis=3, name='conv_tb1_scale1')(trans_x)
    trans_x = Activation('relu')(trans_x)
    conv1 = Conv2D(c1filters, (1, 1),name='conv_tb1_1x1', kernel_initializer='he_uniform', use_bias=False)(trans_x)
    pool1 = AveragePooling2D((2, 2), strides=(2, 2), padding='same',name='avgpool_tb1_2x2')(conv1)
    
    #dense block 2
    dense_x, filters = dense_block(pool1, c1filters, growth_rate, layers_in_block*2, 'db2')
    
    #transition block 2
    c2filters = filters // 2
    trans_x = BatchNormalization(axis=3, name='conv_tb2_bn1')(dense_x)
    trans_x = Scale(axis=3, name='conv_tb2_scale1')(trans_x)
    trans_x = Activation('relu')(trans_x)
    conv2 = Conv2D(c2filters, (1, 1),name='conv_tb2_1x1', kernel_initializer='he_uniform', use_bias=False)(trans_x)
    pool2 = AveragePooling2D((2, 2), strides=(2, 2), padding='same',name='avgpool_tb2_2x2')(conv2)

    #dense block 3
    dense_x, filters = dense_block(pool2, c2filters, growth_rate, layers_in_block*4, 'db3')
    
    #transition block 3
    c3filters = filters // 2
    trans_x = BatchNormalization(axis=3, name='conv_tb3_bn1')(dense_x)
    trans_x = Scale(axis=3, name='conv_tb3_scale1')(trans_x)
    trans_x = Activation('relu')(trans_x)
    conv3 = Conv2D(c3filters, (1, 1),name='conv_tb3_1x1', kernel_initializer='he_uniform', use_bias=False)(trans_x)
    pool3 = AveragePooling2D((2, 2), strides=(2, 2), padding='same',name='avgpool_tb3_2x2')(conv3)

    #dense block 4
    dense_x, filters = dense_block(pool3, c3filters, growth_rate, layers_in_block+10, 'db4')
    
    #skip 1
    up1 =  Conv2DTranspose(c3filters, (2, 2), strides=(2, 2), padding='same', name='up1') (dense_x)
    tb3_up1_conc = concatenate([up1, conv3], axis=-1, name='up1_conv3_tb3')
    
    #depthwise
    dense_x = SeparableConv2D(c3filters*2, (3, 3), name='dw_conv1', padding='same')(tb3_up1_conc)
    dense_x = BatchNormalization(axis=3, name='dw_conv1_bn1')(dense_x)
    dense_x = Scale(axis=3, name='dw_conv1_scale1')(dense_x)
    dense_x = Activation('relu')(dense_x)
    
    #skip 2
    up2 =  Conv2DTranspose(c2filters, (2, 2), strides=(2, 2), padding='same', name='up2') (dense_x)
    tb2_up2_conc = concatenate([up2, conv2], axis=-1, name='up2_conv2_tb2')
    
    #depthwise
    dense_x = SeparableConv2D(c2filters*2, (3, 3), name='dw_conv2', padding='same')(tb2_up2_conc)
    dense_x = BatchNormalization(axis=3, name='dw_conv2_bn2')(dense_x)
    dense_x = Scale(axis=3, name='dw_conv2_scale2')(dense_x)
    dense_x = Activation('relu')(dense_x)
    
    #skip 3
    up3 =  Conv2DTranspose(c1filters, (2, 2), strides=(2, 2), padding='same', name='up3') (dense_x)
    tb1_up3_conc = concatenate([up3, conv1], axis=-1, name='up3_conv1_tb1')
    
    #depthwise
    dense_x = SeparableConv2D(c1filters*2, (3, 3), name='dw_conv3', padding='same')(tb1_up3_conc)
    dense_x = BatchNormalization(axis=3, name='dw_conv3_bn3')(dense_x)
    dense_x = Scale(axis=3, name='dw_conv3_scale3')(dense_x)
    dense_x = Activation('relu')(dense_x)
    
    
    #skip 4
    up4 =  Conv2DTranspose(sfilters, (2, 2), strides=(2, 2), padding='same', name='up4') (dense_x)
    up4_conv = concatenate([up4, conv], axis=-1, name='up4_conv')
    
    #depthwise
    dense_x = SeparableConv2D(sfilters*2, (3, 3), name='dw_conv4', padding='same')(up4_conv)
    dense_x = BatchNormalization(axis=3, name='dw_conv4_bn4')(dense_x)
    dense_x = Scale(axis=3, name='dw_conv4_scale4')(dense_x)
    dense_x = Activation('relu')(dense_x)
    
    up5 =  Conv2DTranspose(sfilters//2, (2, 2), strides=(2, 2), padding='same', name='up5') (dense_x)
    
    #depthwise
    dense_x = SeparableConv2D(sfilters, (3, 3), name='dw_conv5', padding='same')(up5)
    dense_x = BatchNormalization(axis=3, name='dw_conv5_bn5')(dense_x)
    dense_x = Scale(axis=3, name='dw_conv5_scale5')(dense_x)
    dense_x = Activation('relu')(dense_x)
    
    #depthwise
    dense_x = SeparableConv2D(2, (3, 3), name='dw_conv6', padding='same')(dense_x)
    dense_x = BatchNormalization(axis=3, name='dw_conv6_bn6')(dense_x)
    dense_x = Scale(axis=3, name='dw_conv6_scale6')(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = Conv2D(1, (1, 1),name='conv3_1x1', use_bias=False)(dense_x)
    output = Activation('softmax')(dense_x)
    
    return Model(input_img, output)

dense_block_size = 3
layers_in_block = 6
growth_rate = 32
classes = 1
model = dsnet(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
model.compile(Adam(), loss='binary_crossentropy', metrics=[mean_iou])
history = model.fit_generator(
                    train_generator,
                    steps_per_epoch =1894//batch_size, 
                    validation_data=val_generator,
                    validation_steps=350//batch_size,
                    epochs =200 ,
                    callbacks = build_callbacks(),
                    verbose=2)
scores = model.evaluate_generator(test_generator,350//batch_size)
print('loss is '+str(scores[0]))
print('miou is '+str(scores[1]))