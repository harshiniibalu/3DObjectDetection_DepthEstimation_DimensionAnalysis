from ultralytics import YOLOWorld
import cv2
import pyrealsense2 as rs
import numpy as np
model = YOLOWorld("yolov8s-worldv2.pt")
print("YOLO-World model loaded successfully")

model.set_classes(["cardboard box","pen","pencil","eraser","calculator",
                   "mobile phone","book","bottle","cup","mug","toy","apple","watch",
                   "mouse"])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

print("Intel RealSense D455 started successfully")

color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

fx = color_intrinsics.fx
fy = color_intrinsics.fy

print("Focal length X:", fx)
print("Focal length Y:", fy)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            print("Could not capture frame")
            continue
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            box_width = x2 - x1
            box_height = y2 - y1

            # Find the centre of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get depth at the centre point
            distance = depth_frame.get_distance(cx, cy)


            # Calculate real-world dimensions in metres
            real_width = (box_width * distance) / fx
            real_height = (box_height * distance) / fy

            if distance > 0:
                real_width = (box_width * distance) / fx
                real_height = (box_height * distance) / fy

              
                width_cm = real_width * 100
                height_cm = real_height * 100

                label = f"W: {width_cm:.1f} cm H: {height_cm:.1f} cm"
            else:
                label = "Depth unavailable"
                label = f"{distance:.2f} m"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        cv2.imshow("YOLO-World RealSense D455 Detection", annotated_frame)
        cv2.imshow("RealSense Depth Map", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
