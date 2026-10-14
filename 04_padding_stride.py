
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

# Apply convolution with different padding and stride
output1 = F.conv2d(image, kernel, padding=0, stride=1)
output2 = F.conv2d(image, kernel, padding=1, stride=1)
output3 = F.conv2d(image, kernel, padding=0, stride=2)

# Print feature map shapes
print("Input:", image.shape)
print("Padding=0, Stride=1:", output1.shape)
print("Padding=1, Stride=1:", output2.shape)
print("Padding=0, Stride=2:", output3.shape)

# Display results
results = [
    (image, "Original Image"),
    (output1.abs(), "Padding=0, Stride=1"),
    (output2.abs(), "Padding=1, Stride=1"),
    (output3.abs(), "Padding=0, Stride=2")
]

plt.figure(figsize=(12, 4))

for i, (data, title) in enumerate(results, start=1):
    plt.subplot(1, 4, i)
    plt.imshow(data.squeeze().numpy(), cmap="gray", vmin=0)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
