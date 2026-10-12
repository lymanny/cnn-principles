import torch.nn as nn

# Image size: 224 x 224 with 3 RGB channels
height = 224
width = 224
channels = 3

# FC Layer: Connect all input values to 128 neurons
fc = nn.Linear(height * width * channels, 128)

# CNN Layer
# 3 = RGB input channels
# 32 = Number of filters (learn image patterns)
# kernel_size=3 = Each filter is 3x3
cnn = nn.Conv2d(3, 32, kernel_size=3)

# Count FC weights and biases
fc_params = sum(p.numel() for p in fc.parameters())

# Count CNN weights and biases
cnn_params = sum(p.numel() for p in cnn.parameters())

# Show results
print("FC Parameters:", fc_params)
print("CNN Parameters:", cnn_params)