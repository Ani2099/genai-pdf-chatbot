df_tweets = pd.read_csv('/content/Tweets.csv')
print("First 5 rows of the DataFrame:")
print(df_tweets.head())
print("\nDataFrame Information:")
df_tweets.info()

print("Sentiment Distribution:")
print(df_tweets['airline_sentiment'].value_counts())

df_filtered = df_tweets[df_tweets['airline_sentiment'].isin(['positive', 'negative'])].copy()
print(f"Shape of filtered DataFrame: {df_filtered.shape}")
print("Sentiment distribution in filtered DataFrame:")
print(df_filtered['airline_sentiment'].value_counts())

df_binary = df_filtered[['text', 'airline_sentiment']].copy()
print("First 5 rows of df_binary:")
print(df_binary.head())

df_binary['sentiment_label'] = df_binary['airline_sentiment'].map({'positive': 1, 'negative': 0})
print("Sentiment labels added to df_binary:")
print(df_binary[['airline_sentiment', 'sentiment_label']].head())

vocab_size_tweets = 10000
tokenizer_tw = Tokenizer(num_words=vocab_size_tweets, oov_token='<unk>')
tokenizer_tw.fit_on_texts(df_binary['text'])
print(f"Tokenizer fitted with {len(tokenizer_tw.word_index)} unique tokens.")
print(f"Example word index: {list(tokenizer_tw.word_index.items())[:5]}")

sequences = tokenizer_tw.texts_to_sequences(df_binary['text'])
print(f"First 5 sequences: {sequences[:5]}")
print(f"Length of first sequence: {len(sequences[0])}")

maxlen_tweets = 200
padded_sequences = pad_sequences(sequences, maxlen=maxlen_tweets, padding='post')
print(f"Shape of padded sequences: {padded_sequences.shape}")
print(f"First padded sequence: {padded_sequences[0]}")

X = padded_sequences
y = df_binary['sentiment_label'].values
X_train_tw, X_test_tw, y_train_tw, y_test_tw = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train_tw: {X_train_tw.shape}")
print(f"Shape of X_test_tw: {X_test_tw.shape}")
print(f"Shape of y_train_tw: {y_train_tw.shape}")
print(f"Shape of y_test_tw: {y_test_tw.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

model_tw = Sequential()
model_tw.add(Embedding(input_dim=vocab_size_tweets, output_dim=128, input_length=maxlen_tweets))
model_tw.add(GlobalAveragePooling1D())
model_tw.add(Dense(64, activation='relu'))
model_tw.add(Dense(1, activation='sigmoid'))

print("Keras Sequential model 'model_tw' built successfully.")
model_tw.summary()

**Reasoning**:
The previous code block failed because `Sequential`, `Embedding`, `GlobalAveragePooling1D`, and `Dense` were not defined, indicating that the Keras modules were not imported in the execution context. I need to re-import these components from `tensorflow.keras.models` and `tensorflow.keras.layers` before defining the model.



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

model_tw = Sequential()
model_tw.add(Embedding(input_dim=vocab_size_tweets, output_dim=128, input_length=maxlen_tweets))
model_tw.add(GlobalAveragePooling1D())
model_tw.add(Dense(64, activation='relu'))
model_tw.add(Dense(1, activation='sigmoid'))

print("Keras Sequential model 'model_tw' built successfully.")
model_tw.summary()

model_tw.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model 'model_tw' compiled successfully.")

model_tw.fit(X_train_tw, y_train_tw, epochs=10, batch_size=512, validation_data=(X_test_tw, y_test_tw))
print("Model 'model_tw' trained successfully.")

loss_tw, accuracy_tw = model_tw.evaluate(X_test_tw, y_test_tw)
print(f"Test Accuracy for Twitter Sentiment Model: {accuracy_tw * 100:.2f}%")

new_tweets = [
    "@USAirways You guys are the worst! Flight delayed again, service is terrible.",
    "@AmericanAir Great flight today, excellent service and comfortable seats!",
    "@SouthwestAir This flight was okay, nothing special but got me there on time.",
    "@united Never flying with you again, lost my luggage and rude staff.",
    "@Delta Thanks for the smooth travel experience! Always reliable."
]

print("New tweets defined for prediction:")
for i, tweet in enumerate(new_tweets):
    print(f"  {i+1}. {tweet}")

encoded_new_tweets = tokenizer_tw.texts_to_sequences(new_tweets)
print(f"First 2 encoded new tweets: {encoded_new_tweets[:2]}")

padded_new_tweets = pad_sequences(encoded_new_tweets, maxlen=maxlen_tweets, padding='post')
print(f"Shape of padded new tweets: {padded_new_tweets.shape}")
print(f"First padded new tweet: {padded_new_tweets[0]}")

new_predictions = model_tw.predict(padded_new_tweets)
print("Raw predictions for new tweets:")
print(new_predictions)

predicted_labels = (new_predictions > 0.5).astype(int)
print("Predicted binary labels for new tweets:")
print(predicted_labels)

for review, label, score in zip(new_tweets, predicted_labels.flatten(), new_predictions.flatten()):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Review: {review}\nPrediction: {sentiment} (Confidence: {score:.2f})\n")