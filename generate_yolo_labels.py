import os
import cv2
import random
import shutil
from tqdm import tqdm

# ========== SETTINGS ==========
dataset_path = "train"
output_dir = "yolo_dataset"
output_images_dir = os.path.join(output_dir, "images", "train")
output_labels_dir = os.path.join(output_dir, "labels", "train")

# Haarcascade paths
face_cascade_path = "model/haarcascade/haarcascade_frontalface_default.xml"
eye_cascade_path = "model/haarcascade/haarcascade_lefteye_2splits.xml"
mouth_cascade_path = "model/haarcascade/haarcascade_mcs_mouth.xml"

# Class mapping
class_mapping = {
    "Closed": 0,
    "Open": 1,
    "no_yawn": 2,
    "yawn": 3
}

# ========== SETUP ==========
print("🚀 Starting auto-generation of YOLO labels...")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
    raise Exception("❌ Haarcascade models not loaded correctly! Check your model paths.")

# ========== MAIN LOGIC ==========
for cls in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls)
    if not os.path.isdir(cls_path):
        continue

    print(f"📂 Processing class folder: {cls}")

    for img_name in tqdm(os.listdir(cls_path)):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        framegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(framegray, 1.3, 5)

        label_lines = []

        for (x, y, w, h) in faces:
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            label = class_mapping.get(cls, None)
            if label is None:
                continue

            label_lines.append(f"{label} {cx} {cy} {nw} {nh}")

        if label_lines:
            # Save image
            out_img = os.path.join(output_images_dir, img_name)
            cv2.imwrite(out_img, img)

            # Save label
            label_file = os.path.splitext(img_name)[0] + ".txt"
            out_label = os.path.join(output_labels_dir, label_file)
            with open(out_label, "w") as f:
                f.write("\n".join(label_lines))

print("✅ YOLO dataset auto-generation completed successfully!")

# ========== SPLITTING INTO TRAIN/VAL ==========
print("🔁 Splitting dataset into train and val folders...")

# Split ratio
VAL_RATIO = 0.2
image_dir = os.path.join(output_dir, "images", "train")
label_dir = os.path.join(output_dir, "labels", "train")
val_img_dir = os.path.join(output_dir, "images", "val")
val_lbl_dir = os.path.join(output_dir, "labels", "val")

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)
val_count = int(len(image_files) * VAL_RATIO)
val_images = image_files[:val_count]

for img in tqdm(val_images, desc="📦 Moving val files"):
    lbl = os.path.splitext(img)[0] + ".txt"
    shutil.move(os.path.join(image_dir, img), os.path.join(val_img_dir, img))
    lbl_src = os.path.join(label_dir, lbl)
    lbl_dst = os.path.join(val_lbl_dir, lbl)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, lbl_dst)

print("✅ Dataset split into train/val successfully!")
