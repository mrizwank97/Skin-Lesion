import keras
import numpy as np
import tensorflow as tf
from keras import backend as K


def weighted_cross_entropy(y_true, y_pred, beta=1.25):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = tf.log(y_pred / (1 - y_pred))
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
    return tf.reduce_mean(loss)


def balanced_cross_entropy(y_true, y_pred, beta=tf.clip_by_value):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = tf.log(y_pred / (1 - y_pred))
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
    return tf.reduce_mean(loss * (1 - beta))

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.log(y_pred / (1 - y_pred))
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    loss = (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
    return tf.reduce_mean(loss)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    return 1 - numerator / denominator

def tversky_loss(y_true, y_pred, beta=0.5):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

def lovasz_softmax(y_true, y_pred):
    return lovasz_hinge(labels=y_true, logits=y_pred)

