import tensorflow as tf
from tensorflow import keras

import keras_nlp


@keras.utils.register_keras_serializable()
class MySchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, step):
        return tf.cast(self.rate, dtype="float32")

    def get_config(self):
        return {"rate": self.rate}


bert_classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
)
bert_classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=MySchedule(0.001)),
    metrics=["accuracy"],
    jit_compile=True,
)
bert_classifier.save("bert_classifier.keras", save_format="keras_v3")
restored_model = keras.models.load_model("bert_classifier.keras")
