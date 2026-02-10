import torch
from ultralytics import YOLO


def train_model():
    # Auto-detect device: use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {'CUDA GPU' if device == 0 else 'CPU'}")

    # Load base model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        device=device,
        workers=4,
    )


if __name__ == "__main__":
    train_model()
