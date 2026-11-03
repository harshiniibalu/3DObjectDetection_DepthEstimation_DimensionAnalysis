
import cv2
from pathlib import Path

folder = Path("D455_CCA_Data")
video_path = folder / "color_video.mp4"

# Open the saved colour video
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    print("Could not open the colour video.")
    raise SystemExit

# Read the first frame
success, frame = video.read()
video.release()

if not success:
    print("Could not read a frame from the video.")
    raise SystemExit

print("Colour frame shape:", frame.shape)

# Display the frame
cv2.imshow("Recorded Colour Frame", frame)

print("Press any key while the image window is active to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()
