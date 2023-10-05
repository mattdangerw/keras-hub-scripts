from tensorflow import keras
import keras_nlp


classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
)
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(0.5),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
classifier.summary()
classifier.fit(["this is a test", "test test"], y=[1, 0], batch_size=2, epochs=1)
classifier.save("tf-save")
classifier = keras.models.load_model("tf-save")
classifier.save("keras-save.keras", save_format="keras_v3")
classifier = keras.models.load_model("keras-save.keras")
