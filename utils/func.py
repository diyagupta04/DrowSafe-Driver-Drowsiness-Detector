import numpy as np
import cv2

def get_results(results):
    """
    Extract bounding boxes and class labels from YOLO model results.
    """
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    class_ids = results[0].boxes.cls.cpu().tolist()   # Class IDs
    return boxes_xyxy, class_ids

def crop_image(frame, bbox):
    """
    Crop an image based on the provided bounding box coordinates.
    """
    bbox = bbox.astype(int)
    x1, y1, x2, y2 = bbox
    cropped_img = frame[y1:y2, x1:x2]
    return cropped_img

def is_inside(box_small, box_large):
    """
    Check if 'box_small' is completely inside 'box_large'.
    """
    if box_small is not None and box_large is not None:
        return (
            box_small[0] >= box_large[0] and
            box_small[2] <= box_large[2] and
            box_small[1] >= box_large[1] and
            box_small[3] <= box_large[3]
        )
    return False

def calculate_area(bbox):
    """
    Calculate the area of a bounding box.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width * height
