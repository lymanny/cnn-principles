
import torch
import torch.nn as nn

# Create a CNN with 3 Convolution Layers
model = nn.Sequential(
    # Layer 1: 3 RGB Channels → 16 Feature Maps
    # Kernel=3x3, Receptive Field=3x3
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),  # Changes negative values to zero

    # Layer 2: 16 → 32 Feature Maps
    # Kernel=3x3, Receptive Field=5x5
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),

    # Layer 3: 32 → 64 Feature Maps
    # Kernel=3x3, Receptive Field=7x7
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU()
)

# Create a sample RGB image
# Batch=1, Channels=3, Height=224, Width=224
x = torch.randn(1, 3, 224, 224)

# Pass the image through all CNN layers
output = model(x)

# Print important results
print("Input:", x.shape)
print("Output:", output.shape)
print("Final Receptive Field: 7x7")
