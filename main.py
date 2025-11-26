import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

df = pd.read_csv('Language Detection.csv')
X_train,X_test, Y_train,Y_test = train_test_split(df["Text"].astype(str).values,df["Language"].values, test_size=0.2, random_state=42)

le = LabelEncoder()
df["Language"] = le.fit_transform(df["Language"])

#Preprocessing of text

max_tokens = 10000
max_len = 100

vectorize = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",         # "int" = token IDs, "tf-idf" = features
    output_sequence_length=max_len,
    standardize = "lower_and_strip_punctuation"
)
vectorize.adapt(X_train)
# print(X_train[:5])
# print(vectorize.get_vocabulary()[:20]) до этого момента все работает

# Creating a BiLSTM model
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = vectorize(inputs)
x = layers.Embedding(input_dim=max_tokens, output_dim=64, mask_zero=True)(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(len(le.classes_), activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary() #summary of the model



