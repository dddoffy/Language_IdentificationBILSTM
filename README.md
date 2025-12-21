# BiLSTM Language Identification Model (TensorFlow)

### Description
A small NLP project that uses BiLSTM to distinguish between 17 different languages. This model achived 97% accuracy on a validation set. Due to its architecture, this model can work with datasets that contain special symbols, such as commas, dashes, dots, etc. These types of models can identify different word orders, letters, and patterns that are characteristic of a particular language.

## Installation
 ```bash
pip install requirements.txt
```

## Usage

After all the packages are installed, the model will be trained and saved in /artifacts folder. The subsequent runs will use weights, vectorizer and encoder from this folder.
