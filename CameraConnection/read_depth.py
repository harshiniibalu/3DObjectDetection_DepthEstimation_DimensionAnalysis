
import numpy as np
from pathlib import Path

# Load the saved depth frames
folder = Path("D455_CCA_Data")

depth_data = np.load(folder / "depth_frames.npy")

# Read the first captured frame
depth_image = depth_data[0]

# Depth scale obtained from your D455
depth_scale = 0.0010000000474974513

# Select the centre pixel
height, width = depth_image.shape

x = width // 2
y = height // 2

# Read the raw depth value
raw_depth = int(depth_image[y, x])

print("Image width:", width)
print("Image height:", height)
print("Centre pixel coordinates:", (x, y))
print("Raw depth value:", raw_depth)

# Convert to metres if the measurement is valid
if raw_depth == 0 or raw_depth == 65535:
    print("The centre pixel has no valid depth measurement.")
else:
    distance_m = raw_depth * depth_scale

    print("Distance in metres:", distance_m)
    print("Distance in centimetres:", distance_m * 100)