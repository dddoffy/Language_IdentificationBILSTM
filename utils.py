import pickle

def save_label_encoder(le, path):
    with open(path, "wb") as f:
        pickle.dump(le, f)

def load_label_encoder(path):
    with open(path, "rb") as f:
        return pickle.load(f)