import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = tf.keras.models.load_model('best_lstm_model.h5')

max_sequence_length = 100  # match with training

# Function to preprocess input text similar to training
def clean_text(text):
    # Define the same cleaning steps as used before
    import re
    import string
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    # No stopwords removal here for simplicity
    return text

def preprocess(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_sequence_length, padding='post', truncating='post')
    return padded

# Streamlit UI
st.title("Comment Toxicity Detector")

user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        processed = preprocess(user_input)
        prediction = model.predict(processed)[0]
        labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
        st.write("Toxicity Prediction Probabilities:")
        for label, prob in zip(labels, prediction):
            st.write(f"{label}: {prob:.3f}")
