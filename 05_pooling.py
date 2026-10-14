
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load skin image as grayscale
image = Image.open("images/skin.png").convert("L")
image = image.resize((224, 224))

# Convert image to tensor (Batch, Channel, Height, Width)
image = torch.tensor(np.array(image), dtype=torch.float32)
image = image.unsqueeze(0).unsqueeze(0)

# 3x3 vertical edge detection kernel
kernel = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=torch.float32).reshape(1, 1, 3, 3)

# Step 1: Convolution creates a Feature Map
feature_map = F.conv2d(image, kernel, padding=1)

# Step 2: ReLU removes negative values
feature_map = F.relu(feature_map)

# Step 3: Apply 2x2 Max and Average Pooling
max_pool = F.max_pool2d(feature_map, kernel_size=2, stride=2)
avg_pool = F.avg_pool2d(feature_map, kernel_size=2, stride=2)

# Print output shapes
print("Input:", image.shape)
print("Feature Map:", feature_map.shape)
print("Max Pooling:", max_pool.shape)
print("Average Pooling:", avg_pool.shape)

# Display results
results = [
    (image, "Original Image"),
    (feature_map, "Feature Map + ReLU"),
    (max_pool, "Max Pooling"),
    (avg_pool, "Average Pooling")
]

plt.figure(figsize=(12, 4))

for i, (data, title) in enumerate(results, start=1):
    plt.subplot(1, 4, i)
    plt.imshow(data.squeeze().numpy(), cmap="gray", vmin=0)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
