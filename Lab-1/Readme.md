# 🧠 Shallow Architectures with Perceptron & MNIST (TensorFlow/Keras)

This repository contains a hands-on lab that demonstrates how **shallow neural networks** work, starting from the **Perceptron** (the basic building block of neural networks) up to a simple **MNIST digit classifier**.

The project is implemented in **Python** using **Jupyter Notebook in VS Code** (works with Colab too if needed).

---

## 📌 Contents

1. **Perceptron (from scratch with NumPy)**

   - Implementation of a perceptron learning rule.
   - Demonstrates learning for logical functions (AND / OR).

2. **Perceptron (with Keras/TensorFlow)**

   - Single-neuron model with sigmoid activation.
   - Trained on the AND dataset.

3. **Shallow Neural Network on MNIST**

   - Fully connected network:
     - Input: 28×28 flattened digits.
     - Hidden layer (ReLU).
     - Output layer (Softmax for 10 classes).
   - Training, validation, and testing.
   - Evaluation with accuracy, confusion matrix, and error visualization.

4. **Visualization of Misclassifications**
   - Subplots of wrong predictions (`True vs Predicted`).
   - Helps analyze **which digits are confusing the model**.

---
