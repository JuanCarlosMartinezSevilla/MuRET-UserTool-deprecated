import tensorflow as tf
from CRNN.config import Config


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(vocabulary_size):
    input = tf.keras.layers.Input(shape=(Config.img_height, None, 3))
    conv_filters = [16, 32, 64]
    inner = input
    for f in conv_filters:
        inner = tf.keras.layers.Conv2D(f, 3, padding='same')(inner)
        inner = tf.keras.layers.BatchNormalization()(inner)
        inner = tf.keras.layers.LeakyReLU(alpha=0.2)(inner)
        inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inner)

    inner = tf.keras.layers.Permute((2, 1, 3))(inner)
    inner = tf.keras.layers.Reshape(
        target_shape=(-1, (Config.img_height // (2 ** len(conv_filters))) * conv_filters[-1]),
        name='reshape')(inner)

    inner = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(inner)

    inner = tf.keras.layers.Dense(vocabulary_size + 1, name='dense2')(inner)
    y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner)

    model_pr = tf.keras.Model(inputs=input, outputs=y_pred)
    model_pr.summary()

    labels = tf.keras.layers.Input(name='the_labels', shape=[None], dtype='float32')
    input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

    loss_out = tf.keras.layers.Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = tf.keras.Model(inputs=[input, labels, input_length, label_length],
                              outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    return model_tr, model_pr


if __name__ == "__main__":
    pass
