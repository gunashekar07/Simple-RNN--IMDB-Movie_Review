## Step 1: Import Libraries and Load the MOdel
print("App started")

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

print("Libraries loaded")
# Load the IMDB dataset word index
print("Loading IMDB word index...")
word_index=imdb.get_word_index()
print("IMDB word index loaded")

reverse_word_index = { value: key for key, value in word_index.items()}


## Load the Pre-trained model with Relu Activation
print("Loading model...")
model=load_model('simple_rnn_imdb.h5')
print("Model loaded")

model.summary()

# Step 2: Helper Functions 
# Function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review ])

# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

print("🔥 Streamlit starting")

## Streamlit UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
user_input = st.text_area('Enter a review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Score:** {prediction[0][0]:.4f}")