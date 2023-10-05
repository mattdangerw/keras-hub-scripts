import keras_core as keras
# import keras
import tensorflow as tf

tf.debugging.disable_traceback_filtering()
    
batch = tf.ones((2, 1))
inputs = keras.Input(shape=(1,))
outputs = {
    "sum": keras.layers.Add()((inputs, inputs)),
    "diff": keras.layers.Subtract()((inputs, inputs)),
}
model = keras.Model(inputs, outputs)
model.predict(batch)
