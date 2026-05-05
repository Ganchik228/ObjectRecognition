from ultralytics import YOLO
model = YOLO("best.pt")

video_path = "videos/2.mp4"
model.predict(source=video_path, show=True, save=True)
