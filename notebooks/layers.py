# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:06:53 2018

@author: Ryan_ye

refer to https://github.com/SimJeg/FC-DenseNet/blob/master/layers.py
"""
from group_norm import GroupNormalization
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, Dense, BatchNormalization, Input, Reshape, multiply, add, Dropout, AveragePooling2D, GlobalAveragePooling2D, concatenate, Add, Multiply
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model
from keras import backend as k
from keras.regularizers import l2
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils


def GN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)''' 

    l = GroupNormalization(axis=-1, groups=16)(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    name = 'this'
    l = GN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep, attention=False):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    if(attention):
        g1 = Conv2D(n_filters_keep, kernel_size = 1)(skip_connection) 
        g1 = BatchNormalization()(g1)
        x1 = Conv2D(n_filters_keep,kernel_size = 1)(l) 
        x1 = GroupNormalization(axis=-1, groups=16)(x1)
        g1_x1 = Add()([g1,x1])
        psi = Activation('relu')(g1_x1)
        psi = Conv2D(1,kernel_size = 1)(psi) 
        #psi = GroupNormalization(axis=-1, groups=16)(psi)
        psi = Activation('sigmoid')(psi)
        skip_connection = Multiply()([skip_connection,psi])
    l = concatenate([l, skip_connection], axis=-1)
    return l

def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(inputs)
#    l = Reshape((-1, n_classes))(l)
    l = Activation('sigmoid')(l)#or softmax for multi-class
    return l
    
    
