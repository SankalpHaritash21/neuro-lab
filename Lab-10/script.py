# sentiment_lstm_imdb.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load IMDB (top 10k words) ---
num_words = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test  = pad_sequences(x_test,  maxlen=maxlen)

# --- Model ---
model = models.Sequential([
    layers.Embedding(num_words, 128, input_length=maxlen),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train (2 epochs is often enough for demo) ---
model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1)

# --- Evaluate ---
loss, acc = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test acc: {acc:.4f}")

# --- Example prediction (requires index_to_word) ---
word_index = imdb.get_word_index()
index_to_word = {i+3:w for w,i in word_index.items()}
index_to_word[0] = '<pad>'; index_to_word[1] = '<start>'; index_to_word[2] = '<unk>'

def decode_review(seq):
    return ' '.join(index_to_word.get(i, '?') for i in seq)

sample_idx = 10
print("Sample review (decoded):", decode_review(x_test[sample_idx]))
print("Predicted sentiment:", model.predict(x_test[sample_idx:sample_idx+1])[0,0])
