import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import numpy as np

# Load model
model = tf.keras.models.load_model('next_word_model.h5')

# Load tokenizer (if saved)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

#maximum sequence length
with open('config.json', 'r') as f:
    config = json.load(f)

max_seq_len = config['max_seq_len']

# Function to predict the next word(s)
def predict_next_word(seed_text, n_words=1):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]

        # Try to get the next word
        next_word = tokenizer.index_word.get(predicted_word_index)

        if next_word is None:
            seed_text += " [UNKNOWN]"
            break  # Or continue to try next words, depending on your logic
        else:
            seed_text += " " + next_word
    return seed_text

# Streamlit UI
st.title("🧠 Next Word Predictor")
st.write("Enter a small phrase and predict the next word(s) using an LSTM model.")


# User input
user_input = st.text_input("Enter a text prompt:")
num_words = st.slider("Number of words to predict:", 1, 10, 1)

if st.button("Predict Next Word(s)"):
    if user_input.strip():
        predicted_text = predict_next_word(user_input, n_words=num_words)
        st.subheader("Predicted text:")
        st.success(predicted_text)
    else:
        st.warning("Please enter some text to begin.")