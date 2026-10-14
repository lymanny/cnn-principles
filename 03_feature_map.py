
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load skin image as grayscale
image = Image.open("images/skin.png").convert("L")

# Convert image to tensor (Batch, Channel, Height, Width)
image = torch.tensor(np.array(image), dtype=torch.float32)
image = image.unsqueeze(0).unsqueeze(0)

# Filter 1: Vertical edges
vertical = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

# Filter 2: Horizontal edges
horizontal = [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
]

# Combine 2 filters (2, 1, 3, 3)
kernels = torch.tensor(
    [vertical, horizontal],
    dtype=torch.float32
).unsqueeze(1)

# Apply convolution to produce 2 feature maps
feature_maps = F.conv2d(image, kernels)

# Display original image and feature maps
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image.squeeze().numpy(), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(feature_maps[0, 0].abs().numpy(), cmap="gray", vmin=0)
plt.title("Feature Map 1 - Vertical Edges")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(feature_maps[0, 1].abs().numpy(), cmap="gray", vmin=0)
plt.title("Feature Map 2 - Horizontal Edges")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print shapes
print("Input Shape:", image.shape)
print("Filters Shape:", kernels.shape)
print("Feature Maps Shape:", feature_maps.shape)
