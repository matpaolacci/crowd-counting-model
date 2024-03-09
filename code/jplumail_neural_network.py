"""The model here was taken from https://github.com/jplumail/people-counting/blob/main/train.py"""
import tensorflow as tf

def count(densityMap):
    return tf.math.reduce_sum(densityMap, axis=(1,2))

def mse_count(densityMapTrue, densityMapPred):
    return tf.math.reduce_mean(((count(densityMapTrue) - count(densityMapPred)) ** 2))

def mae_count(densityMapTrue, densityMapPred):
    return tf.math.reduce_mean(tf.math.abs(count(densityMapTrue) - count(densityMapPred)))

def getNormalizationLayer(data):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
    normalization_layer.adapt(data)
    return normalization_layer

def buildJplumailCcnn(learningRate, inputHeight, inputWidth, preprocessingLayer):
    im = tf.keras.layers.Input(shape=(inputHeight, inputWidth, 3))
    out = preprocessingLayer(im)
    out1 = tf.keras.layers.Conv2D(10, 9, padding='same')(out)
    out2 = tf.keras.layers.Conv2D(14, 7, padding='same')(out)
    out3 = tf.keras.layers.Conv2D(16, 5, padding='same')(out)
    features = tf.keras.layers.Concatenate()([out1, out2, out3])
    
    layers = tf.keras.models.Sequential()
    layers.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    layers.add(tf.keras.layers.Conv2D(60, 3, padding='same'))
    layers.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    layers.add(tf.keras.layers.Conv2D(40, 3, padding='same'))
    layers.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    layers.add(tf.keras.layers.Conv2D(20, 3, padding='same'))
    layers.add(tf.keras.layers.Conv2D(10, 3, padding='same'))
    layers.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    layers.add(tf.keras.layers.UpSampling2D((8,8), interpolation='bilinear'))
 
    out = layers(features)
    ccnn = tf.keras.Model(inputs=im, outputs=out, name="C-CNN")
 
    ccnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss="MSE", metrics=[mae_count, mse_count])
    return ccnn