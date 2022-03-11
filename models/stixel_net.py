#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow as tf

def build_stixel_net(input_shape=(1280, 1920, 3)):
    """
    input_shape -> (height, width, channel)
    """
    img_input = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding="same")(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(2048, (3, 1), strides=(1, 1), padding="valid")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 1), strides=(1, 1))(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(1, (1, 1), strides=(1, 1))(x)

    x = layers.Reshape((240, 1))(x)

    model = models.Model(inputs=img_input, outputs=x)

    return model




def build_stixel_net_small(input_shape=(1280, 1920, 3)):
    """
    input_shape -> (height, width, channel)
    """
    img_input = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding="same")(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(64, (3, 1), strides=(1, 1), padding="valid")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(64, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(64, (1, 1), strides=(1, 1))(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(1, (1, 1), strides=(1, 1))(x)

    x = layers.Reshape((240, 1))(x)

    model = models.Model(inputs=img_input, outputs=x)

    return model



def build_stixel_net_inceptionV3(input_shape=(1280,1920,3)):
    
    img_input = keras.Input(shape=input_shape)

    target_shape = (320,480,3)

    # pretrained net
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=target_shape,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False

    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

    x = layers.experimental.preprocessing.Resizing(target_shape[0], target_shape[1])(img_input)

    x = preprocess_input(x)
    x = base_model(x, training=False)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(240)(x)   

    model = models.Model(inputs=img_input, outputs=x)

    return model