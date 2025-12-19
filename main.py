import os
from train import train_model
from predict import predict_language
import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    text = str(input("Enter text: "))

    if not os.path.exists("artifacts/language_model.keras"):
        print("No model found. Training a new model...")
        train_model()
    else:
        print("Model found. Skipping training.")

    language = predict_language(text)
    print(f"Predicted language: {language}")

if __name__ == "__main__":
    main()