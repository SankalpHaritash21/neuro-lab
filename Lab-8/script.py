# text_language_model_rnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Example corpus (replace with your larger corpus) ---
texts = [
    "the quick brown fox jumps over the lazy dog",
    "the quick blue fox leaps over the sleepy cat",
    "this is a simple language model example",
    "language models can predict next words"
]

# --- Tokenize and build sequences ---
tok = Tokenizer(oov_token='<OOV>')
tok.fit_on_texts(texts)
vocab_size = len(tok.word_index) + 1
seqs = tok.texts_to_sequences(texts)

pairs_X = []
pairs_y = []
for s in seqs:
    for i in range(1, len(s)):
        pairs_X.append(s[:i])
        pairs_y.append(s[i])

maxlen = max(len(x) for x in pairs_X)
X = pad_sequences(pairs_X, maxlen=maxlen, padding='pre')
y = tf.keras.utils.to_categorical(pairs_y, num_classes=vocab_size)

# --- Model ---
embed_dim = 64
rnn_units = 128
model = models.Sequential([
    layers.Embedding(vocab_size, embed_dim, input_length=maxlen),
    layers.SimpleRNN(rnn_units, return_sequences=False),
    layers.Dense(vocab_size, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train (small epochs because corpus is tiny) ---
model.fit(X, y, epochs=200, verbose=0)

# --- Sampling / predict next word given seed text ---
def predict_next(seed_text, k=3):
    seq = tok.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=maxlen, padding='pre')
    probs = model.predict(seq, verbose=0)[0]
    top_k = np.argsort(probs)[-k:][::-1]
    return [(tok.index_word.get(i, '<OOV>'), float(probs[i])) for i in top_k]

# Example:
print("Seed: 'the quick'")
print(predict_next("the quick", k=5))
