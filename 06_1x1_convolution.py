
import torch
import torch.nn as nn

# Input: Batch=1, Feature Maps=64, Height=224, Width=224
x = torch.randn(1, 64, 224, 224)

# 1x1 Convolution: Reduce 64 → 16 Feature Maps
conv_reduce = nn.Conv2d(64, 16, kernel_size=1)
output_reduce = conv_reduce(x)

# 1x1 Convolution: Increase 64 → 128 Feature Maps
conv_increase = nn.Conv2d(64, 128, kernel_size=1)
output_increase = conv_increase(x)

# 1x1 Convolution: Keep 64 Feature Maps
conv_same = nn.Conv2d(64, 64, kernel_size=1)
output_same = conv_same(x)

# Print results
print("Input:", x.shape)
print("Reduce:", output_reduce.shape)
print("Increase:", output_increase.shape)
print("Same:", output_same.shape)
print("Kernel Size:", conv_reduce.kernel_size)
