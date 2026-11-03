
import numpy as np
import cv2
from pathlib import Path

# Locate the saved depth data
folder = Path("D455_CCA_Data")
file_path = folder / "depth_frames.npy"

# Load the saved data
depth_data = np.load(file_path)

print("Depth data loaded successfully!")
print("Shape:", depth_data.shape)
print("Data type:", depth_data.dtype)

# Select the first captured frame
depth_image = depth_data[0]

# Find the minimum and maximum non-zero depth values
valid_depth = depth_image[depth_image > 0]

if valid_depth.size == 0:
    print("No valid depth measurements in this frame.")
else:
    print("Minimum raw depth:", valid_depth.min())
    print("Maximum raw depth:", valid_depth.max())

    # Create a display-only preview
    preview = cv2.normalize(
        depth_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Convert the grayscale preview to a colour map
    colored_preview = cv2.applyColorMap(
        preview,
        cv2.COLORMAP_JET
    )

    # Save the preview image
    output_path = folder / "depth_preview.png"

    cv2.imwrite(str(output_path), colored_preview)

    print("Depth preview saved to:", output_path.resolve())

