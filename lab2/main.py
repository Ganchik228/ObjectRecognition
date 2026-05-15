import matlab.engine
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

OBJECTNESS_DIR = r"C:\Users\User\Documents\Learn\ObjectRecognition\lab2\objectness-release-v2.2"
IMAGE_PATH = r"C:\Users\User\Documents\Learn\ObjectRecognition\lab2\images\5.jpg"
CAT_CLASS_IDS = [281, 282, 283, 284, 285]



def yolo_has_cat(yolo_model, crop):
    results = yolo_model(crop, verbose=False, classes=CAT_CLASS_IDS)
    return any(len(r.probs.top5) > 0 for r in results)


eng = matlab.engine.start_matlab()
eng.addpath(OBJECTNESS_DIR, nargout=0)
eng.addpath(OBJECTNESS_DIR + r"\MEX", nargout=0)
eng.addpath(OBJECTNESS_DIR + r"\pff_segment", nargout=0)
eng.cd(OBJECTNESS_DIR, nargout=0)
eng.startup(nargout=0)

matlab_path = IMAGE_PATH.replace("\\", "/")
boxes = np.array(eng.eval(f"runObjectness(imread('{matlab_path}'), 10)", nargout=1))
eng.quit()

positive_boxes = boxes[boxes[:, 4] >= 0.5]
negative_boxes = boxes[boxes[:, 4] <  0.5]


yolo = YOLO("yolo26n-cls.pt")
img = cv2.imread(IMAGE_PATH)
img_h, img_w = img.shape[:2]


def classify_boxes(yolo_model, img, box_list):
    results = []
    for box in box_list:
        x1 = max(0, int(box[0]));  y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2])); y2 = min(img_h, int(box[3]))
        if x2 - x1 < 10 or y2 - y1 < 10:
            results.append(False)
            continue
        results.append(yolo_has_cat(yolo_model, img[y1:y2, x1:x2]))
    return results


positive_results = classify_boxes(yolo, img, positive_boxes)

negative_results = classify_boxes(yolo, img, negative_boxes)


TP = sum(1 for r in positive_results if r)
FN = sum(1 for r in positive_results if not r) 

FP = sum(1 for r in negative_results if r)
TN = sum(1 for r in negative_results if not r)

P = TP + FN
N = FP + TN
acc = (TP + TN) / (P + N) * 100 if (P + N) > 0 else 0

print(f"""
╔══════════════════════════════════════╗
║          │ Predicted + │ Predicted - ║
╠══════════════════════════════════════╣
║ Actual + │  TP = {TP:>4}  │  FN = {FN:>4}  ║
║ Actual - │  FP = {FP:>4}  │  TN = {TN:>4}  ║
╠══════════════════════════════════════╣
║ P={P:<4}  N={N:<4}  Accuracy = {acc:.1f}%     ║
╚══════════════════════════════════════╝
""")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img_rgb)

for i, box in enumerate(positive_boxes):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = 'deepskyblue' if positive_results[i] else 'red'  # TP или FN
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, lw=2, edgecolor=color, facecolor='none'
    ))

ax.set_title(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}  |  Accuracy={acc:.1f}%")
ax.legend(handles=[
    patches.Patch(color='deepskyblue', label=f'TP = {TP} (предложен + кот найден)'),
    patches.Patch(color='red',         label=f'FN = {FN} (предложен + кота нет)'),
])
plt.tight_layout()
plt.savefig("result.png", dpi=100)
plt.show()
