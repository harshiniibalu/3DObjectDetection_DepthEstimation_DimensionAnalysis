from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo26n.pt")
print("YOLO model loaded successfully")

#accessing the laptop web cam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
else:
    print("Webcam opened successfully!")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not capture frame")
        break

    results = model(frame,conf = 0.50, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()