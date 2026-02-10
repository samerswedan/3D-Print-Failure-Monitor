import os
from ultralytics import YOLO

# Updated path based on your successful 'train3' run
model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model not found at {model_path}")
    print("Please check your 'runs/detect' folder for the correct 'train' number.")
else:
    print(f"✅ Found model. Starting NCNN export...")
    try:
        model = YOLO(model_path)

        # Exporting to NCNN
        # 'half=True' uses FP16, which is essential for Pi 4 performance
        model.export(format="ncnn", half=True)

        print("\n🚀 SUCCESS!")
        print(f"Your NCNN model folder is in the same directory as your best.pt")
    except Exception as e:
        print(f"❌ Export failed: {e}")
