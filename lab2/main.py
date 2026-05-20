import matlab.engine
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

OBJECTNESS_DIR = r"C:\Users\User\Documents\Learn\ObjectRecognition\lab2\objectness-release-v2.2"
IMAGE_PATH = r"C:\Users\User\Documents\Learn\ObjectRecognition\lab2\images\kitty.jpg"

CAT_CLASS_ID = 17 


model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

def has_cat(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(crop_rgb)
    
    output = model([tensor])[0]
    
    for label, score in zip(output["labels"], output["scores"]):
        if label.item() == CAT_CLASS_ID and score.item() >= 0.3:
            return True
    return False


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


img = cv2.imread(IMAGE_PATH)
print(img.shape)
img_h, img_w = img.shape[:2]

def classify_boxes(img, box_list):
    results = []
    for box in box_list:
        x1 = int(box[0])  
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        results.append(has_cat(img[y1:y2, x1:x2]))
    return results

positive_results = classify_boxes(img, positive_boxes)
negative_results = classify_boxes(img, negative_boxes)

TP = sum(1 for r in positive_results if r)
TN = sum(1 for r in positive_results if not r)
FP = sum(1 for r in negative_results if r)
FN = sum(1 for r in negative_results if not r)

P = TP + FN
N = FP + TN
acc = (TP + TN) / (P + N)

print(f"""
╔══════════════════════════════════════╗
║          │ Predicted + │ Predicted - ║
╠══════════════════════════════════════╣
║ Actual + │  TP = {TP}  │  FN = {FN}  ║
║ Actual - │  FP = {FP}  │  TN = {TN}  ║
╠══════════════════════════════════════╣
║ P={P}  N={N}  Accuracy = {acc}       ║
╚══════════════════════════════════════╝
""")


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img_rgb)

for i, box in enumerate(positive_boxes):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = 'deepskyblue' if positive_results[i] else 'red'
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, lw=2, edgecolor=color, facecolor='none'
    ))

ax.set_title(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}  |  Accuracy={acc}")
ax.legend(handles=[
    patches.Patch(color='deepskyblue', label=f'TP = {TP}'),
    patches.Patch(color='red', label=f'FN = {FN}'),
])
plt.tight_layout()
plt.savefig("result.png", dpi=100)
plt.show()