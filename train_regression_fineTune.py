#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np

from config import _CURRENT_DIR
sys.path.append(os.path.join(_CURRENT_DIR, "."))

try:
    from config import Config
    from data_loader import WaymoStixelDataset
except:
    print("failed to load module")
    
from albumentations import (
    Resize,
    Compose,
    CLAHE,
    HueSaturationValue,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    ToFloat,
    Normalize,
    GaussNoise,
    RandomShadow,
    RandomRain,
)

import utility
import importlib
from models.stixel_net import build_stixel_net_inceptionV3, build_stixel_net_small, build_stixel_net

# TensorFlow
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)


# In[2]:


dt_config = Config()
dt_config.display()


# In[3]:


## Train and Val set


train_aug = Compose(
    [
        GaussNoise(p=1.0),
        RandomShadow(p=0.5),
        RandomRain(p=0.5, rain_type="drizzle"),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        CLAHE(p=0.5, clip_limit=2.0),
        Normalize(p=1.0),
    ]
)
    
# AUGMENTATIONS DISABLED
train_set = WaymoStixelDataset(
        data_path=dt_config.DATA_PATH,
        ground_truth_path=os.path.join(dt_config.DATA_PATH, "waymo_train.txt"),
        batch_size=4,
        transform=None,
        customized_transform=utility.HorizontalFlip(p=0.5),
    )




val_aug = Compose([Normalize(p=1.0)])
val_set = WaymoStixelDataset(
        data_path=dt_config.DATA_PATH,
        ground_truth_path=os.path.join(dt_config.DATA_PATH, "waymo_val.txt"),
        transform=None,
    )


# In[4]:


def custom_loss(y_actual,y_pred):
 
    mask = tf.cast(tf.math.less(y_actual, tf.constant([2.0])), dtype=tf.float32)    
    custom_loss=tf.reduce_mean(tf.math.abs( tf.math.multiply( (y_actual-y_pred), mask)), axis=-1)  
    return custom_loss


# In[5]:


# StixelNet regression setup

#model = build_stixel_net_inceptionV3()
model = build_stixel_net()
opt = optimizers.Adam(0.0001)
#lossF = losses.MeanSquaredError()

model.compile(loss=custom_loss, optimizer=opt)
model.summary()


# In[6]:


X,y = train_set[0]
print(np.shape(X))
print(np.shape(y))


# In[7]:


# Training

num_epoch = 1000

callbacks = [
    ModelCheckpoint(
        os.path.join(dt_config.SAVED_MODELS_PATH, "model-{epoch:03d}-{loss:.4f}.h5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=False,
        mode="auto",
        save_freq="epoch",
    ),    
    EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="auto"
    ),
]


history = model.fit(
        train_set,       
        validation_data=val_set,      
        epochs=num_epoch,
        callbacks=callbacks,
        shuffle=True,
    )

