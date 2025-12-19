import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import build_model
import sys

sys.stdout.reconfigure(encoding='utf-8')
ENCODER_PATH = "artifacts/label_encoder.pkl"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"


def train_model():
    df = pd.read_csv("data/Language Detection.csv", encoding="utf-8")
    le = LabelEncoder()
    df["Language"] = le.fit_transform(df["Language"])

    x_train, x_test, y_train, y_test = train_test_split(
        df["Text"].astype(str).values,
        df["Language"].values,
        test_size=0.2,
        random_state=42
    )

    model, vectorizer = build_model(num_classes=len(le.classes_))

    #Adapt vectorizer to text
    vectorizer.adapt(x_train)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump({
            'config': vectorizer.get_config(),
            'vocab': vectorizer.get_vocabulary()
        }, f)

    #Convert text to numbers manually before training
    x_train_seq = vectorizer(x_train)

    #Train on the numbers
    model.fit(x_train_seq, y_train, epochs=5, batch_size=32)

    model.save("artifacts/language_model.keras")
    pickle.dump(le, open(ENCODER_PATH, "wb"))

    print("✅ Model trained and saved.")