import keras_core as keras
import numpy as np

@keras.saving.register_keras_serializable()
class Reverse(keras.layers.Layer):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def call(self, x):
        return keras.ops.matmul(x, keras.ops.transpose(self.kernel))


@keras.saving.register_keras_serializable()
class Model(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(10)
        self.dense.build((None, 5))
        self.reverse = Reverse(self.dense.kernel)

    def call(self, inputs):
        return self.reverse(self.dense(inputs))

    def get_config(self):
        return {}


model = Model()
model(np.ones((8, 5)))
model.save("test.keras")

restored = keras.models.load_model("test.keras")
print(restored.weights)
