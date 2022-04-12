import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, Model, optimizers

class SymbolCNN:

    def get_conv_block(input_img, filters, kernel, pooling_kernel):
        x = input_img
        for f in filters:
            x = layers.Conv2D(f, kernel, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(pooling_kernel, strides=(2,2))(x)
            x = layers.Dropout(0.25)(x)
        
        return x

    def model(cats, i_h, i_w):
        #
        # if its glyph --> 40 x 40
        #
        # if its position --> 40 x 112

        filters = [32, 32, 64, 64]
        kernel = (3 , 3)
        pooling_kernel = (2,2)

        input_img = layers.Input(shape=(i_h, i_w, 1))

        #input_img_glyph = layers.Input(shape=(40, 40, 1))
        #input_img_pos   = layers.Input(shape=(112, 40, 1))
        
        x = SymbolCNN.get_conv_block(input_img, filters, kernel, pooling_kernel)
        #x_glyph = SymbolCNN.get_conv_block(input_img_glyph, filters, kernel, pooling_kernel)
        #x_pos   = SymbolCNN.get_conv_block(input_img_pos, filters, kernel, pooling_kernel)

        x = layers.Flatten()(x) # (bs, h/8, w/8, 64) -> (bs, f) -- f = h/8*w/8*64
        #x_glyph = layers.Flatten()(x_glyph) # (bs, h/8, w/8, 64) -> (bs, f) -- f = h/8*w/8*64
        #x_pos   = layers.Flatten()(x_pos)   # (bs, h/8, w/8, 64) -> (bs, f) -- f = h/8*w/8*64

        #x_concat = layers.Concatenate(axis=1)([x_glyph, x_pos])

        x = layers.Dense(256)(x)

        #x_ff = layers.Dense(256)(x_concat)

        out  = layers.Dense(cats,  activation="softmax", name='Dense')(x)
        #out_glyph  = layers.Dense(cats_glyph,  activation="softmax", name='Glyphs_Dense')(x_ff)
        #out_pos    = layers.Dense(cats_pos,    activation="softmax", name='Positions_Dense')(x_ff)

        model = Model(inputs=[input_img], outputs=[out])

        # To obtain 2 models that its weights are modified when training with 2 inp and 2 outp
        #model_g = Model(inputs=[input_img_glyph], outputs=[out_glyph])
        #model_p = Model(inputs=[input_img_pos], outputs=[out_pos])

        #opt = optimizers.rmsprop_v2()

        model.compile(optimizer='RMSprop', loss= losses.SparseCategoricalCrossentropy())
        #return model,  model_g, model_p
        return model