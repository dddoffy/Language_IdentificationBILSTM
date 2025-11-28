import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

df = pd.read_csv('Language Detection.csv')


le = LabelEncoder()
df["Language"] = le.fit_transform(df["Language"])
X = df["Text"].astype(str).values
y = df["Language"].values


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


max_tokens = 10000
max_len = 100

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",         # "int" = token IDs, "tf-idf" = features
    output_sequence_length=max_len,
    standardize = "lower_and_strip_punctuation"
)
vectorizer.adapt(X_train)


# Creating a BiLSTM model
inputs = layers.Input(shape=(1,), dtype=tf.string) # gets a string of text
x = vectorizer(inputs) # vectorizes this string
x = layers.Embedding(input_dim=max_tokens, output_dim=64, mask_zero=True)(x) # takes each integer token converts it to dense vector
x = layers.Bidirectional(layers.LSTM(64))(x) # bidirectional lstm 128; dimensional output
x = layers.Dense(64, activation="relu")(x) #h learns high level features
outputs = layers.Dense(len(le.classes_), activation="softmax")(x) # gives a probability of a language

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary() #summary of the model


history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=32
)

loss, acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", acc)

sample = tf.constant(["Bonjour, comment vas-tu ?"])
pred = model.predict(sample)
lang_id = pred.argmax(axis=1)[0]

print("Predicted language:", le.inverse_transform([lang_id])[0])




