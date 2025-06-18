import cv2
import time
import numpy as np
import pygame
from threading import Thread
from ultralytics import YOLO
from tensorflow.keras.utils import img_to_array
from utils.config import project_config as pj
from utils.func import get_results, crop_image, calculate_area, is_inside


class DrowsinessDetector:
    def __init__(self, clf_model) -> None:
        self.clf_model = clf_model
        self.class_labels = ["Closed", "Open", "yawn", "no_yawn"]

        self.eye_status = []
        self.yawn_detected = False
        self.start_time = 0
        self.count_start = False
        self.time_close_eyes = 0
        self.alarm_on = False

        # Load YOLO model directly
        self.yolo_model = YOLO(pj.YOLO_MODEL_PATH)
        print("✅ YOLO model loaded successfully!")

    def preprocess_eye_frame(self, eye_frame):
        try:
            eye_frame = cv2.resize(eye_frame, (145, 145))
            eye_frame = eye_frame.astype("float32") / 255.0
            eye_frame = img_to_array(eye_frame)
            eye_frame = np.expand_dims(eye_frame, axis=0)
            return eye_frame
        except Exception as e:
            print(f"Error preprocessing eye frame: {e}")
            return None

    def start_alarm(self, sound_path):
        pygame.mixer.init()
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()

    def detect_drowsiness(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model.predict(frame_rgb, conf=0.6, verbose=False)

        # Plot YOLO detections
        frame = results[0].plot()
        boxes, classes = get_results(results)

        head_boxes = []
        eye_boxes = []
        mouth_boxes = []

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            if cls == 1:  # Head
                head_boxes.append(box)
            elif cls == 0:  # Eye
                eye_boxes.append(box)
            elif cls == 2:  # Mouth
                mouth_boxes.append(box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Class {cls}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Select largest head
        largest_head = max(head_boxes, key=calculate_area, default=None)

        self.eye_status.clear()
        self.yawn_detected = False

        if largest_head:
            for eye_box in eye_boxes:
                if is_inside(eye_box, largest_head):
                    eye_crop = crop_image(frame, eye_box)
                    processed_eye = self.preprocess_eye_frame(eye_crop)
                    if processed_eye is not None:
                        prediction = self.clf_model.predict(processed_eye, verbose=0)
                        label = self.class_labels[np.argmax(prediction)]
                        self.eye_status.append(label)

            for mouth_box in mouth_boxes:
                if is_inside(mouth_box, largest_head):
                    self.yawn_detected = True
                    break

        closed_eyes_count = self.eye_status.count("Closed")
        open_eyes_count = self.eye_status.count("Open")

        if (closed_eyes_count >= 2) or self.yawn_detected:
            if not self.count_start:
                self.start_time = time.time()
                self.count_start = True
            else:
                self.time_close_eyes = time.time() - self.start_time

            if self.time_close_eyes >= pj.TIME_THRESHOLD or self.yawn_detected:
                if not self.alarm_on:
                    self.alarm_on = True
                    Thread(target=self.start_alarm, args=(pj.ALARM_SOUND,), daemon=True).start()

                alert_text = "Yawn Detected!" if self.yawn_detected else "Drowsiness Alert!"
                cv2.putText(frame, alert_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, pj.RED, 3)
            else:
                cv2.putText(frame, f"Eyes Closed: {round(self.time_close_eyes, 2)}s",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, pj.RED, 3)
        elif open_eyes_count >= 2:
            self.reset_counters()
            cv2.putText(frame, "Eyes Open", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, pj.GREEN, 3)

        return frame

    def reset_counters(self):
        self.count_start = False
        self.time_close_eyes = 0
        self.alarm_on = False
