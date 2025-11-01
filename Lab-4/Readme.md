# Simple CNN: Filtering → Convolution → Pooling (PyTorch + TensorFlow)

### Simple CNN Teaching Demo

This repo contains **two small demos** to explain how Convolutional Neural Networks (CNNs) work, step by step:

1. **Filtering & Manual Convolution (`filtering_demo_short.py`)**

   - Shows how a 3×3 kernel slides over an image (with elementwise multiplication and summation).
   - Demonstrates common filters: blur, sharpen, edge detection.
   - Includes pooling example.

2. **PyTorch CNN Demo (`pytorch_conv_demo_short.py`)**
   - Creates a Conv2D layer with **hand-crafted kernels** (sharpen, blur, edge).
   - Applies them to a grayscale image.
   - Visualizes outputs after **convolution → ReLU → pooling**.

---

## What we learned:

1. How Convolution Works (Arithmetic)

   - A convolution is basically:
     Slide a small filter (kernel) → multiply elementwise with a patch of the image → add up the results → put into the feature map.

   - You saw this manually with the 6×6 image and 3×3 kernel.
     Example: top-left 3×3 patch + kernel multiplication gave 25.
     That’s exactly what the computer does across the whole image.

You now understand the arithmetic behind each number in the feature map.

2. Pooling (Downsampling)

   - Pooling reduces the size of the feature map.
   - In max pooling, we take the maximum value in a block (e.g., 2×2).
   - This keeps the strongest features but shrinks dimensions → less computation, more robustness.

   You learned why feature maps shrink (4×4 → 2×2) and why CNNs can detect patterns regardless of exact location.

3. PyTorch Implementation

   - How to build a Conv2d layer with custom weights.
   - You saw how to set the kernel in PyTorch manually and visualize:
     - Input image
     - Kernel
     - Feature map
     - Pooled map

   You learned how deep learning libraries actually implement the math you did manually.

4. TensorFlow Implementation

   - Similar to PyTorch, but TensorFlow expects data in a different shape (NHWC instead of NCHW).
   - You learned how to set weights, run a forward pass, and pool.

   You now understand that frameworks differ in details but do the same underlying math.

5. Conceptual Understanding

   - CNN = Filtering + Non-linearity + Pooling, repeated in layers.
   - Each filter extracts different features: edges, textures, patterns.
   - Pooling helps generalize and reduce complexity.
   - Visualization shows how images transform step by step.

   You now know the building blocks of CNNs and can explain them to others.

6. Skills Gained

   - Write NumPy code for manual convolution and pooling.
   - Use PyTorch/TensorFlow layers for convolution and pooling.
   - Visualize how feature maps evolve.
   - Explain CNNs from arithmetic → code → visual intuition.

We learned how CNNs actually work at the lowest level (math) and how to implement them in both PyTorch and TensorFlow, plus why pooling is used.
