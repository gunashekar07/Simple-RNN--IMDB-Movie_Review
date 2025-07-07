## Step 1: Import Libraries and Load the MOdel

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index=imdb.get_word_index()
word_index
reverse_word_index = { value: key for key, value in word_index.items()}
reverse_word_index

## Load the Pre-trained model with Relu Activation

model=load_model('simple_rnn_imdb.h5')
model.summary()

# Step 2: Helper Functions 
# Function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review ])

# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



import Streamlit as st
## streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

## User Input
user_input= st.text_area('Movie Review')
if st.button('Classify'):
    ## Make Prediction
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    ## display the result
    st.write(f'sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')