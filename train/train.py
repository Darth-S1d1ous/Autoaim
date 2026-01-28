from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data=os.path.abspath("data/data.yaml"),
        epochs=100,
        imgsz=960,
        device=0,
        workers=4,
        optimizer="AdamW",
        weight_decay=5e-4,
        warmup_epochs=3,
        close_mosaic=10
    )

if __name__ == "__main__":
    main()