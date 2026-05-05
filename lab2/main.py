import cv2
import numpy as np
import os

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing import image

# --- настройки ---
IMAGE_DIR = "images"
MODEL_PATH = "ObjectnessTrainedModel"
TARGET_CLASS = "cat"  # <-- поменяй под свою задачу
MAX_BOXES = 50

# --- загрузка CNN ---
model = MobileNetV2(weights="imagenet")

# --- BING ---
bing = cv2.saliency.ObjectnessBING_create()
bing.setTrainingPath(os.path.abspath(MODEL_PATH))

total_TP = 0
total_FP = 0

# --- обработка всех изображений ---
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    print(f"\n=== {img_name} ===")

    # 1. гипотезы
    success, boxes = bing.computeSaliency(img)

    if not success or boxes is None:
        print("Ошибка BING")
        continue

    print("Всего гипотез:", len(boxes))

    boxes = boxes[:MAX_BOXES]
    print("Используем:", len(boxes))

    TP = 0
    FP = 0

    # 2. классификация
    for (x, y, w, h) in boxes:
        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        try:
            crop_resized = cv2.resize(crop, (224, 224))
        except:
            continue

        x_input = image.img_to_array(crop_resized)
        x_input = np.expand_dims(x_input, axis=0)
        x_input = preprocess_input(x_input)

        preds = model.predict(x_input, verbose=0)
        label = decode_predictions(preds, top=1)[0][0][1]

        if TARGET_CLASS in label:
            TP += 1
        else:
            FP += 1

    print("TP:", TP)
    print("FP:", FP)

    total_TP += TP
    total_FP += FP

# --- итог ---
TN = 0
FN = 0

total = total_TP + total_FP + TN + FN
accuracy = (total_TP + TN) / total if total > 0 else 0

print("\n=== ИТОГ ===")
print("TP:", total_TP)
print("FP:", total_FP)
print("TN:", TN)
print("FN:", FN)
print("Accuracy:", accuracy)