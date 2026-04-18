# NLP Text Classification using Neural Network (Keras)

## Step 1: Import Libraries
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.model_selection import train_test_split

## Step 2: Load Dataset (IMDB)
from tensorflow.keras.datasets import imdb # Import the imdb dataset

vocab_size = 10000  # Top 10,000 words
maxlen = 200  # Max length of review
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

## Step 3: Pad Sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

## Step 4: Build Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

## Step 5: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Step 6: Train Model
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_test, y_test))

## Step 7: Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load word index used by IMDB
word_index = imdb.get_word_index()

# Add offset of 3 because 0, 1, 2 are reserved (padding, start, unknown)
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Create reverse index for debug if needed
reverse_word_index = {value: key for key, value in word_index.items()}

# Function to convert text to sequence
def encode_review(text):
    tokens = text.lower().split()
    return [1] + [word_index.get(word, 2) for word in tokens]  # 1 = <START>, 2 = <UNK>

# Sample live reviews
live_reviews = [
    "This movie was an absolute masterpiece with incredible performances.",
    "Worst movie ever. I wasted two hours of my life.",
    "Brilliant direction and storyline. I loved every minute of it!",
    "Terrible script, poor acting. Would not recommend.",
    "The film had some good moments, but overall it was boring."
]

# Encode and pad
encoded_reviews = [encode_review(review) for review in live_reviews]
padded_reviews = pad_sequences(encoded_reviews, maxlen=200, padding='post')

# Predict (assuming 'model' is your trained model)
predictions = model.predict(padded_reviews)
labels = (predictions > 0.5).astype(int)

# Display results
for review, label, score in zip(live_reviews, labels, predictions.flatten()):
    sentiment = "Positive" if label else "Negative"
    print(f"Review: {review}\nPrediction: {sentiment} (Confidence: {score:.2f})\n")