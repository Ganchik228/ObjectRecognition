from ultralytics import YOLO
model = YOLO("yolo26n-cls.pt")
print(model.names)