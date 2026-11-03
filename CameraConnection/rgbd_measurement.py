
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

# Folder containing your recorded data
folder = Path("D455_CCA_Data")

depth_path = folder / "depth_frames.npy"
video_path = folder / "color_video.mp4"

# Check that the required files exist
if not depth_path.exists() or not video_path.exists():
    raise FileNotFoundError(
        "Required files were not found in D455_CCA_Data."
    )

# Load the recorded depth frames
depth_frames = np.load(depth_path)

# Open the recorded colour video
video = cv2.VideoCapture(str(video_path))

success, color_frame = video.read()
video.release()

if not success:
    raise RuntimeError("Could not read the colour video.")

# Load the camera depth scale
# This matches the scale reported by your D455.
depth_scale = 0.0010000000474974513

print("Colour image shape:", color_frame.shape)
print("Depth frame shape:", depth_frames[0].shape)

# The RGB image and depth image must correspond to the
# same captured moment and must be aligned.
# This program assumes that the saved colour and depth
# frames are already aligned and correspond to each other.
print("\nSelect the rectangular box using your mouse.")
print("Press ENTER or SPACE to confirm your selection.")
print("Press C to cancel.")

# Select the box in the colour image
selection = cv2.selectROI(
    "Select the box",
    color_frame,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = map(int, selection)

if w == 0 or h == 0:
    raise RuntimeError("No object region was selected.")

# Select the corresponding depth frame.
# This uses the first depth frame for now.
depth_image = depth_frames[0]

# Confirm image dimensions match.
if depth_image.shape != color_frame.shape[:2]:
    raise RuntimeError(
        "Colour and depth dimensions do not match. "
        "Alignment must be checked before combining them."
    )

# Extract depth measurements in the selected region
region = depth_image[y:y+h, x:x+w]

# Ignore invalid depth measurements.
# The value 65535 was observed in your earlier data;
# zero is also an invalid depth value.
valid = (
    (region > 0)
    & (region < 65535)
)

valid_depths = region[valid]

if len(valid_depths) == 0:
    raise RuntimeError(
        "No valid depth measurements found in the selected region."
    )

# Use the median to reduce the influence of noisy pixels.
median_raw = np.median(valid_depths)
median_depth_m = median_raw * depth_scale
median_depth_cm = median_depth_m * 100

print("\n--- Object depth result ---")
print("Selected region:", (x, y, w, h))
print("Valid depth pixels:", len(valid_depths))
print(f"Median distance: {median_depth_m:.3f} metres")
print(f"Median distance: {median_depth_cm:.2f} centimetres")

# Convert colour image from OpenCV's BGR order to RGB.
rgb_image = cv2.cvtColor(
    color_frame,
    cv2.COLOR_BGR2RGB
)

# Convert raw depth values to metres.
depth_metres = depth_image.astype(np.float32) * depth_scale

# Build a four-channel RGB-D array.
# This is only valid if RGB and depth are properly aligned.
rgbd_array = np.dstack((rgb_image, depth_metres))

# Save the RGB-D array.
array_path = folder / "rgbd_array.npy"
np.save(array_path, rgbd_array)

# Create a CSV table for pixels inside the selected box region.
yy, xx = np.mgrid[y:y+h, x:x+w]

rgb_region = rgb_image[y:y+h, x:x+w]
depth_region_m = depth_metres[y:y+h, x:x+w]

valid_region = (
    (region > 0)
    & (region < 65535)
)

table = np.column_stack((
    xx[valid_region],
    yy[valid_region],
    rgb_region[valid_region],
    depth_region_m[valid_region]
))

csv_path = folder / "box_rgbd_matrix.csv"

np.savetxt(
    csv_path,
    table,
    delimiter=",",
    header="X,Y,R,G,B,Depth_m",
    comments="",
    fmt=["%d", "%d", "%d", "%d", "%d", "%.6f"]
)

print("\nFiles saved:")
print(array_path.resolve())
print(csv_path.resolve())

print("\nIMPORTANT:")
print(
    "Verify RGB-depth alignment and frame synchronization "
    "before treating these results as valid measurements."
)
