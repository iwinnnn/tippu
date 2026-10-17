
import cv2
from getpass import getpass
from urllib.parse import quote
from ultralytics import YOLO

# --- MODULE 1: CAMERA INGESTION ---


ip = "10.146.44.219"
username = "tapo1234"
password = "12345678"

url = (
    f"rtsp://{quote(username, safe='')}:"
    f"{quote(password, safe='')}@{ip}:554/stream1"
)

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Camera connection failed.")
    raise SystemExit(1)

# --- MODULE 2: PERSON DETECTION ---

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

print("AI detection started. Press Q to quit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Camera frame unavailable.")
            break

        # COCO class 0 = person
        results = model.predict(
            source=frame,
            classes=[0],
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()
        person_count = len(result.boxes)

        cv2.putText(
            annotated_frame,
            f"People detected: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Campus Space Identifier - AI Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
