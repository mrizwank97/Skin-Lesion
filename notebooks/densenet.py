import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Input, Activation, Dropout, GlobalAveragePooling2D, BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
from custom_layers import Scale

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


def transition_block(trans_x, tran_filters, name):
    trans_x = BatchNormalization(axis=3, name='conv_'+name+'_bn1')(trans_x)
    trans_x = Scale(axis=3, name='conv_'+name+'_scale1')(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1),name='conv_'+name+'_1x1', kernel_initializer='he_uniform', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2),name='avgpool_'+name+'_2x2')(trans_x)

    return trans_x, tran_filters


def model(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_img = Input(shape=(256, 256, 3))
  
    dense_x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(input_img)
    dense_x = Conv2D(filters, (7, 7), strides=(2,2), name='conv1', kernel_initializer='he_uniform', use_bias=False)(dense_x)
    dense_x = BatchNormalization(axis=3, name='conv1_bn')(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = Scale(axis=3, name='conv1_scale')(dense_x)
    dense_x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(dense_x)
    dense_x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(dense_x)
    
    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block,'db1')
    dense_x, filters = transition_block(dense_x, filters//2, 'tb1')

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block*2, 'db2')
    dense_x, filters = transition_block(dense_x, filters//2, 'tb2')
    
    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block*4, 'db3')
    dense_x, filters = transition_block(dense_x, filters//2, 'tb3')
    
    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block+10, 'db4')
    
    dense_x = BatchNormalization(axis=3, name='conv2_bn2')(dense_x)
    dense_x = Scale(axis=3, name='conv2_scale')(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(classes, activation='softmax')(dense_x)

    return Model(input_img, output)