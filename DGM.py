# DGM.py (TensorFlow 2.x Compatible)

import tensorflow as tf

def _to_dtype(dtype):
    # Accept "float32" / "float64" or a tf.DType
    if isinstance(dtype, str):
        return tf.float64 if dtype.lower() == "float64" else tf.float32
    return tf.as_dtype(dtype)

class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh", dtype=tf.float32):
        super().__init__(dtype=_to_dtype(dtype))
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.trans1 = getattr(tf.nn, trans1) if isinstance(trans1, str) else trans1
        self.trans2 = getattr(tf.nn, trans2) if isinstance(trans2, str) else trans2

        init = tf.keras.initializers.GlorotUniform()
        d = self.dtype  # layer dtype

        self.Uz = self.add_weight(name="Uz", shape=[input_dim, output_dim], initializer=init, dtype=d)
        self.Ug = self.add_weight(name="Ug", shape=[input_dim, output_dim], initializer=init, dtype=d)
        self.Ur = self.add_weight(name="Ur", shape=[input_dim, output_dim], initializer=init, dtype=d)
        self.Uh = self.add_weight(name="Uh", shape=[input_dim, output_dim], initializer=init, dtype=d)

        self.Wz = self.add_weight(name="Wz", shape=[output_dim, output_dim], initializer=init, dtype=d)
        self.Wg = self.add_weight(name="Wg", shape=[output_dim, output_dim], initializer=init, dtype=d)
        self.Wr = self.add_weight(name="Wr", shape=[output_dim, output_dim], initializer=init, dtype=d)
        self.Wh = self.add_weight(name="Wh", shape=[output_dim, output_dim], initializer=init, dtype=d)

        self.bz = self.add_weight(name="bz", shape=[1, output_dim], initializer="zeros", dtype=d)
        self.bg = self.add_weight(name="bg", shape=[1, output_dim], initializer="zeros", dtype=d)
        self.br = self.add_weight(name="br", shape=[1, output_dim], initializer="zeros", dtype=d)
        self.bh = self.add_weight(name="bh", shape=[1, output_dim], initializer="zeros", dtype=d)

    def call(self, S, X):
        # Ensure inputs are in-layer dtype
        S = tf.cast(S, self.dtype)
        X = tf.cast(X, self.dtype)

        Z = self.trans1(tf.matmul(X, self.Uz) + tf.matmul(S, self.Wz) + self.bz)
        G = self.trans1(tf.matmul(X, self.Ug) + tf.matmul(S, self.Wg) + self.bg)
        R = self.trans1(tf.matmul(X, self.Ur) + tf.matmul(S, self.Wr) + self.br)

        H = self.trans2(tf.matmul(X, self.Uh) + tf.matmul(R * S, self.Wh) + self.bh)
        S_new = (1 - G) * H + Z * S
        return S_new


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim, transformation=None, dtype=tf.float32):
        super().__init__(dtype=_to_dtype(dtype))
        init = tf.keras.initializers.GlorotUniform()
        d = self.dtype

        self.W = self.add_weight(name="W", shape=[input_dim, output_dim], initializer=init, dtype=d)
        self.b = self.add_weight(name="b", shape=[1, output_dim], initializer="zeros", dtype=d)

        if transformation:
            self.transformation = getattr(tf.nn, transformation) if isinstance(transformation, str) else transformation
        else:
            self.transformation = None

    def call(self, X):
        X = tf.cast(X, self.dtype)
        S = tf.matmul(X, self.W) + self.b
        if self.transformation is not None:
            S = self.transformation(S)
        return S


class DGMNet(tf.keras.Model):
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None, dtype=tf.float32):
        # Set model-level dtype; child layers inherit passed dtype explicitly
        super().__init__(dtype=_to_dtype(dtype))
        d = self.dtype
        in_dim = input_dim + 1  # time t concatenated with x

        self.initial_layer = DenseLayer(layer_width, in_dim, transformation="tanh", dtype=d)
        self.LSTMLayerList = [LSTMLayer(layer_width, in_dim, dtype=d) for _ in range(n_layers)]
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans, dtype=d)

    def call(self, t, x):
        # Cast inputs to the model dtype and concatenate
        t = tf.cast(t, self.dtype)
        x = tf.cast(x, self.dtype)
        X = tf.concat([t, x], axis=1)
        S = self.initial_layer(X)
        for lstm in self.LSTMLayerList:
            S = lstm(S, X)
        return self.final_layer(S)
