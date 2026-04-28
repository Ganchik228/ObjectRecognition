from ultralytics import YOLO

if __name__ == '__main__':
    # Новая модель с нуля
    model_new = YOLO("yolo11n.yaml")
    model_new.train(data="coco8.yaml", epochs=100, imgsz=640)

    # Предобученная без дообучения
    model_pretrained = YOLO("yolo11n.pt")

    # Предобученная с дообучение
    model_finetune = YOLO("yolo11n.pt")
    model_finetune.train(data="coco8.yaml", epochs=100, imgsz=640)
   
   
    
    metrics1 = model_new.val(data="coco8.yaml")
    metrics2 = model_pretrained.val(data="coco8.yaml")
    metrics3 = model_finetune.val(data="coco8.yaml")

    print(f"Новая с нуля:                  mAP50 = {metrics1.box.map50:.3f}")
    print(f"Предобученная без дообучения:  mAP50 = {metrics2.box.map50:.3f}")
    print(f"Предобученная + дообучение:    mAP50 = {metrics3.box.map50:.3f}")

