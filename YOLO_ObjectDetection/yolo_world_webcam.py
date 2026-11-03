from ultralytics import YOLOWorld
import numpy as np 
import cv2

model = YOLOWorld("yolov8s-worldv2.pt")
print("YOLO-World model loaded successfully")

model.set_classes(["cardboard box","pen","pencil","eraser","calculator",
                   "mobile phone","book","bottle","cup","mug","toy","person","apple","watch","mouse"])
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not capture frame")
        break

    results = model(frame,  verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO-World Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()