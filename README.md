# 🧠 CNN Principles

A beginner-friendly repository covering **7 fundamental CNN principles** with practical Python and PyTorch examples.

## 📚 Topics

### 1. 🧮 Problems with Fully Connected (FC) Layers

Learn why Fully Connected Layers require many parameters when processing images.

Topics:

- Fully Connected Layers
- Parameter Counting
- FC vs. CNN Comparison
- Local Connectivity and Weight Sharing

### 2. 🔍 Convolution Operation

Learn how convolution kernels extract visual features from images.

Topics:

- Convolution Operation
- Kernel Size
- Local Connectivity
- Weight Sharing
- Edge Detection

### 3. 🗺️ Feature Maps

Learn how Convolution Layers generate Feature Maps.

Topics:

- Feature Map Generation
- Vertical and Horizontal Filters
- Multiple Filters
- Feature Map Visualization
- Channels and Feature Maps

### 4. 📐 Padding and Stride

Learn how Padding and Stride control convolution output dimensions.

Topics:

- Zero Padding
- Stride
- Kernel Movement
- Feature Map Dimensions
- Output Size Calculation

### 5. 🧩 Pooling

Learn how Pooling reduces the spatial size of Feature Maps.

Topics:

- Max Pooling
- Average Pooling
- Pooling Kernel Size
- Pooling Stride
- Downsampling

### 6. 🔢 1×1 Convolution

Learn how 1×1 Convolution changes the number of Feature Maps.

Topics:

- 1×1 Kernel
- Input and Output Channels
- Feature Map Reduction
- Feature Map Expansion
- Channel Preservation

### 7. 🎯 Receptive Field

Learn how the Receptive Field grows through multiple Convolution Layers.

Topics:

- Receptive Field
- Multiple Convolution Layers
- Kernel Size vs. Receptive Field
- Receptive Field Growth
- Deeper Feature Extraction

## 🔄 CNN Process

A simplified CNN feature-extraction process:

```
Input Image
    ↓
Convolution
    ↓
Feature Maps
    ↓
ReLU Activation
    ↓
Pooling
    ↓
Smaller Feature Maps
    ↓
Additional Convolution Layers
    ↓
Higher-Level Features
```

**Notes:**

- **Padding & Stride:** Control convolution behavior and output size.
- **Pooling:** Reduces the spatial dimensions of Feature Maps.
- **1×1 Convolution:** Changes the number of Feature Maps.
- **Receptive Field:** Increases through multiple Convolution Layers.

These concepts are related CNN principles, not seven mandatory sequential stages.

**Note:** Some examples use images from the `images/` folder. A GPU is not required for these beginner exercises.

## 📖 References

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)
- [PyTorch — Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/)
- [PyTorch — Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch — MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
- [PyTorch — AvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
- [Stanford CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Dive into Deep Learning — Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
