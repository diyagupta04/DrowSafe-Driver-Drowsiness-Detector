from pydantic import BaseSettings

class ProjectConfig(BaseSettings):
    # Frame size for webcam input
    WIDTH: int = 1280
    HEIGHT: int = 720
    SIZE: tuple = (1280, 720)

    # Basic colors (BGR format for OpenCV)
    BLUE: tuple = (255, 0, 0)
    GREEN: tuple = (0, 255, 0)
    RED: tuple = (0, 0, 255)
    WHITE: tuple = (255, 255, 255)

    # Drowsiness detection settings
    TIME_THRESHOLD: int = 2  # Time (in seconds) eyes must be closed or yawn detected
    ALARM_SOUND: str = "public/alarm.wav"  # Path to alarm sound

    # YOLO Model Path (for face, eyes, mouth detection)
    YOLO_MODEL_PATH: str = "model/yolov8n.pt"  # Update if you fine-tune a custom YOLO model

    # Trained CNN Model for eye (open/closed) and mouth (yawn/no-yawn) classification
    CNN_MODEL_PATH: str = "model/CNN_model.keras"

# Instantiate the config
project_config = ProjectConfig()
