
import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import time
from pathlib import Path

# 1. Create a folder for the saved data
output_folder = Path("D455_CCA_Data")
output_folder.mkdir(exist_ok=True)

# 2. Create the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Request depth and color streams
config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

# Prepare variables
video_writer = None
depth_frames = []
distance_records = []

try:
    # 3. Connect to the camera and start streaming
    profile = pipeline.start(config)

    # Get the scale needed to convert raw depth units to metres
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("D455 connected successfully.")
    print("Depth scale:", depth_scale, "metres per unit")
    print("Recording for 10 seconds...")

    # 4. Allow camera exposure to settle
    for _ in range(15):
        pipeline.wait_for_frames()

    start_time = time.time()
    frame_number = 0

    # 5. Capture data for 10 seconds
    while time.time() - start_time < 10:

        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert the camera frames into NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save the raw depth values in memory
        depth_frames.append(depth_image.copy())

        # Create an MP4 video from the color frames
        if video_writer is None:
            video_path = str(output_folder / "color_video.mp4")

            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (color_image.shape[1], color_image.shape[0])
            )

            if not video_writer.isOpened():
                raise RuntimeError(
                    "Could not create MP4. Check OpenCV/video codec support."
                )

        video_writer.write(color_image)

        # Measure distance at the centre of the depth image
        center_x = depth_frame.get_width() // 2
        center_y = depth_frame.get_height() // 2

        distance_m = depth_frame.get_distance(center_x, center_y)

        timestamp = time.time() - start_time

        distance_records.append([
            frame_number,
            round(timestamp, 4),
            round(distance_m, 5)
        ])

        frame_number += 1

    # 6. Save numerical depth data to a file
    if depth_frames:
        depth_array = np.stack(depth_frames, axis=0)

        np.save(
            output_folder / "depth_frames.npy",
            depth_array
        )

    # Save the centre-point distance readings
    with open(
        output_folder / "center_distance.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame_number",
            "timestamp_seconds",
            "center_distance_metres"
        ])

        writer.writerows(distance_records)

    # Save the depth scale
    with open(output_folder / "depth_scale.txt", "w") as file:
        file.write(f"Depth scale in metres per unit: {depth_scale}\n")

    print("Recording complete.")
    print("Frames captured:", frame_number)
    print("Saved files in:", output_folder.resolve())

except Exception as error:
    print("Camera capture error:", error)

finally:
    if video_writer is not None:
        video_writer.release()

    pipeline.stop()
    print("Camera stopped.")