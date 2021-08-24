import tensorflow as tf

OPTIMIZER_DICT = {
    "adam": tf.keras.optimizers.Adam(1e-2),
    "sgd": tf.keras.optimizers.SGD(1e-2),
    "rmsprop": tf.keras.optimizers.RMSprop(1e-2),
}
