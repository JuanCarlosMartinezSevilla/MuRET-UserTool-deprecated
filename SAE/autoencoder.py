from SAE.daConfig import DAConfig
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model

def getModel():
    img_tam = DAConfig.image_size
    
    input_img = Input(shape=(img_tam, img_tam, 1))

    x = input_img
    
    filters = DAConfig.filters
    kernel = DAConfig.kernel
    pool = DAConfig.pool

    for idx, f in enumerate(filters):
        if idx <= 2:
            x = Conv2D(f, kernel, activation='relu', padding='same')(x)
            x = MaxPooling2D(pool)(x)
        else:
            x = Conv2D(f, kernel, activation='relu', padding='same')(x)
            x = UpSampling2D(pool)(x)

    decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

    SAEmodel = Model(input_img, decoded)
    SAEmodel.compile(optimizer='adam', loss = 'binary_crossentropy')

    return SAEmodel

        