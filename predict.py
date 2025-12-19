import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import sys

sys.stdout.reconfigure(encoding='utf-8')

ENCODER_PATH = "artifacts/label_encoder.pkl"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"


def predict_language(text):

    model = load_model("artifacts/language_model.keras")


    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        data = pickle.load(f)

    vectorizer = TextVectorization.from_config(data['config'])


    vectorizer.set_vocabulary(data['vocab'])

    #Convert text to numbers
    text_seq = vectorizer([text])


    pred = model.predict(text_seq)
    lang_id = pred.argmax(axis=1)[0]

    return le.inverse_transform([lang_id])[0]