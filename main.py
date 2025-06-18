import cv2
import time
import threading
from ultralytics import YOLO
from utils.config import project_config as pj
import pygame

# Load YOLOv8 model
yolo_model = YOLO("runs/detect/train7/weights/best.pt")
pygame.mixer.init()

# Alarm control
alarm_lock = threading.Lock()
alarm_playing = False
yawn_start_time = None  # Track when yawn starts

def play_alarm():
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            return
        alarm_playing = True
    try:
        pygame.mixer.music.load("public/alarm.wav")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Alarm error: {e}")
    with alarm_lock:
        alarm_playing = False

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, pj.WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pj.HEIGHT)

class_names = ['Closed', 'Open', 'no_yawn', 'yawn']
num_frames = 0
start_time = time.time()
yawn_detected = False
yawn_start_time = None

print("🚀 Real-time detection started. Press 'q' or ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, imgsz=640, conf=0.1)[0]
    current_time = time.time()
    yawn_present = False

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = class_names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(frame, (x1, y1), (x2, y2), pj.GREEN, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pj.GREEN, 2)

        if label == 'yawn':
            yawn_present = True

    # Handle yawn duration logic
    if yawn_present:
        if yawn_start_time is None:
            yawn_start_time = current_time
        elif current_time - yawn_start_time >= 3:  # Yawn persists for 3 seconds
            threading.Thread(target=play_alarm, daemon=True).start()
    else:
        yawn_start_time = None  # Reset if no yawn

    # Show FPS
    num_frames += 1
    fps = num_frames / (current_time - start_time)
    cv2.putText(frame, f"FPS: {round(fps, 1)}", (10, pj.HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pj.GREEN, 2)
    cv2.imshow("YOLOv8 Yawn Detector", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Finished. Camera released.")
