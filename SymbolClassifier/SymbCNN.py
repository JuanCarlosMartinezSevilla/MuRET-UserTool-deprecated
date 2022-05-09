import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, Model
from SymbolClassifier.configuration import Configuration

class SymbolCNN:

    def get_conv_block(input_img, filters, kernel, pooling_kernel):
        x = input_img
        for f in filters:
            x = layers.Conv2D(f, kernel, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(pooling_kernel, strides=(2,2))(x)
            x = layers.Dropout(0.25)(x)
        
        return x

    def model(cats, i_h, i_w):

        filters = Configuration.filters
        kernel = Configuration.kernel
        pooling_kernel = Configuration.pooling_kernel

        input_img = layers.Input(shape=(i_h, i_w, 1))
        
        x = SymbolCNN.get_conv_block(input_img, filters, kernel, pooling_kernel)

        x = layers.Flatten()(x) # (bs, h/8, w/8, 64) -> (bs, f) -- f = h/8*w/8*64

        x = layers.Dense(256)(x)

        out  = layers.Dense(cats,  activation="softmax", name='Dense')(x)
        
        model = Model(inputs=[input_img], outputs=[out])

        model.compile(optimizer='RMSprop', loss= losses.CategoricalCrossentropy())
        
        return model