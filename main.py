import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
vocab_size = 10000
maxlen = 200

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build and train the model
st.write("⏳ Training model (this takes about 1-2 minutes)...")

model = Sequential([
    Embedding(vocab_size, 32, input_length=maxlen),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train[:5000], y_train[:5000], epochs=2, batch_size=64, verbose=1)

st.success("✅ Model trained and ready!")

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis with SimpleRNN")
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    # Convert text to sequence (simple tokenizer based on IMDB word_index)
    word_index = imdb.get_word_index()
    tokens = [word_index.get(word.lower(), 2) for word in user_input.split()]
    seq = pad_sequences([tokens], maxlen=maxlen)
    prediction = model.predict(seq)[0][0]

    sentiment = "Positive 😀" if prediction > 0.5 else "Negative 😞"
    st.write(f"**Prediction:** {sentiment}")
    st.write(f"Confidence: {prediction:.2f}")
