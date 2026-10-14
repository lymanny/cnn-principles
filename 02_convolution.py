
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load skin image
image = Image.open("images/skin.png").convert("L")
image = torch.tensor(np.array(image), dtype=torch.float32)
image = image.unsqueeze(0).unsqueeze(0)

# Vertical edge kernel
vertical = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=torch.float32).reshape(1, 1, 3, 3)

# Horizontal edge kernel
horizontal = torch.tensor([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
], dtype=torch.float32).reshape(1, 1, 3, 3)

# Apply convolution
v_edges = F.conv2d(image, vertical, padding=1).abs()
h_edges = F.conv2d(image, horizontal, padding=1).abs()

# Combine vertical and horizontal edges
edges = torch.sqrt(v_edges**2 + h_edges**2)

# Display results
plt.figure(figsize=(12, 4))

for i, (data, title) in enumerate([
    (image, "Original"),
    (v_edges, "Vertical Edges"),
    (h_edges, "Horizontal Edges"),
    (edges, "Combined Edges")
], start=1):
    plt.subplot(1, 4, i)
    plt.imshow(data.squeeze().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
