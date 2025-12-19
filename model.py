import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization


def build_model(num_classes, max_tokens=10000, max_len=100,):

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation"
    )

    #Creating a BiLSTM model
    inputs = layers.Input(shape=(max_len,), dtype=tf.int64) #gets a string of text
    x = layers.Embedding(input_dim=max_tokens, output_dim=64, mask_zero=True)(
        inputs)  #takes each integer token converts it to dense vector
    x = layers.Bidirectional(layers.LSTM(64))(x)  #bidirectional lstm 128; dimensional output
    x = layers.Dense(64, activation="relu")(x)  #h learns high level features
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, vectorizer


