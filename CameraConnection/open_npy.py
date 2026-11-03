
import numpy as np
from pathlib import Path

# Locate the saved RGB-D matrix
file_path = Path("D455_CCA_Data") / "rgbd_array.npy"

# Load the file
data = np.load(file_path)

# Display information
print("RGB-D matrix loaded successfully!")
print("Shape:", data.shape)
print("Data type:", data.dtype)

# Read one pixel
x = 320
y = 240

print("\nPixel coordinates:", (x, y))
print("Red:", data[y, x, 0])
print("Green:", data[y, x, 1])
print("Blue:", data[y, x, 2])
print("Depth in metres:", data[y, x, 3])
print("Depth in centimetres:", data[y, x, 3] * 100)
