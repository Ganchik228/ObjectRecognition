from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy

model = YOLO("best.pt") #yolov8n.pt

video_path = "videos/1.mp4"
output_path = "output/1_output.mp4"

# model.train(data="data.yaml", epochs=5, imgsz=1280)

tracker = DeepSort(max_age=150, n_init=3, max_cosine_distance=0.7, nn_budget=None)

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра моделью YOLO
    results = model(frame, conf=0.6, iou=0.5)

    detections = []
    for result in results:
        if len(result.boxes.xyxy) > 0:
            # Преобразуем tензор в numpy для удобства
            boxes = result.boxes.xyxy.cpu().numpy()  # Преобразуем в numpy массив
            conf = float(result.boxes.conf.cpu().numpy()[0])  # Уверенность модели
            cls = int(result.boxes.cls.cpu().numpy()[0])  # Класс объекта
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                # Формат для трекера: [x1, y1, width, height, confidence, class_id]
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

    # Обновляем трекер
    tracks = tracker.update_tracks(detections, frame=frame)

    # Отображаем треки на кадре
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Рисуем прямоугольник и ID трека
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Coin ID {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Сохраняем обработанный кадр
    out.write(frame)

cap.release()
out.release()
print(f"Обработка завершена, результат сохранён в {output_path}")