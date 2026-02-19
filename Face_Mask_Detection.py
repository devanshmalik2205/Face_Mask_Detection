import cv2
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "models/Face_Mask_Detection_v1.pt"

try:
    model = YOLO(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    logging.error("Could not open webcam.")
    exit()

logging.info("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
