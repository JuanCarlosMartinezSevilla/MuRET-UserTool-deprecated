from tf.keras.layers import Input, Conv2D, Flatten, Dense
from tf.keras.layers import MaxPooling2D, UpSampling2D, Dropout
from tf.keras.models import Model

class SymbolCNN:

    def model(img_heigth, img_width, epochs, generator, steps):
        #
        # if its glyph --> 40 x 40
        #
        # if its position --> 40 x 112

        filters = [32, 32, 64, 64]
        kernel = (3,3)
        pooling_kernel = (2,2)

        input_img = Input(shape=(img_heigth, img_width, 1))
        
        for f in filters:
            x = Conv2D(f, kernel, activation='relu', padding='same')(input_img)
            x = MaxPooling2D(pooling_kernel)(x)
            x = Dropout(0.25)(x)

        x = Flatten()(x)
        # aquí hay que hacer un concat
        x = Dense(256, activation='softmax')(x)
        output = Dropout(0.25)(x)
        # fork models

        SymbolCNNmodel = Model(input_img, output)
        


