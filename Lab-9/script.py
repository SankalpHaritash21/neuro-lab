# time_series_rnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Generate data: sine wave ---
t = np.linspace(0, 100, 10001)
data = np.sin(t)  # shape (10001,)

# --- Build sequences (sliding window) ---
seq_len = 50
X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len])
X = np.array(X)[..., np.newaxis]  # (N, seq_len, 1)
y = np.array(y)                   # (N,)

# --- Train/val split ---
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# --- Model ---
model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(seq_len,1)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Train ---
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# --- One-step prediction demonstration ---
i = -200  # some index in validation range
pred = model.predict(X_val[i:i+1])[0,0]
print(f"True next value: {y_val[i]:.4f}, Predicted: {pred:.4f}")

# Optional: plot prediction vs true for a short span
preds = model.predict(X_val[:200]).squeeze()
plt.plot(y_val[:200], label='True')
plt.plot(preds, label='Pred')
plt.legend(); plt.title("Short prediction (validation)"); plt.show()
