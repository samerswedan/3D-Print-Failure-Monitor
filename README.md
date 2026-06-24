# 3D Print Failure Detection

A real-time 3D print monitoring system using YOLOv8 and computer vision to detect print failures (spaghetti, stringing, warping) and send Discord alerts.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-time Detection**: Monitors your 3D printer via webcam and detects failures as they happen
- **Multi-class Detection**: Identifies spaghetti, stringing, and warping failures
- **Discord Alerts**: Sends instant notifications with failure snapshots to your Discord server
- **Temporal Filtering**: Uses a consistency buffer to reduce false positives
- **Web Dashboard**: Streamlit-based UI with live video feed and status logs
- **Headless Mode**: CLI version (`main.py`) for running without a browser

## Requirements

- Python 3.10+
- Webcam connected to your system

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/samerswedan/3DPrintFailML.git
   cd 3DPrintFailML
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Discord alerts** (optional)

   ```bash
   cp .env.example .env
   # Edit .env and add your Discord webhook URL
   ```

## ⚙️ Configuration

### Discord Webhook Setup

1. Open your Discord server settings
2. Go to **Integrations** → **Webhooks**
3. Click **New Webhook** and copy the URL
4. Add the URL to your `.env` file:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

## Usage

### Web Dashboard (Recommended)

```bash
streamlit run app.py
```

Opens a browser dashboard with:

- Live camera feed with AI annotations
- Adjustable confidence threshold
- Configurable consistency buffer
- Real-time status and alert logs

### CLI Mode (Headless)

```bash
python main.py
```

Displays an OpenCV window with the camera feed. Press `q` to quit.

## 🐳 Docker

The Streamlit dashboard can be run in a container. The image is CPU-only
(no CUDA), runs as a non-root user, and exposes the dashboard on port `8501`.

### Using Docker Compose (recommended)

```bash
# Configure your Discord webhook first
cp .env.example .env   # then edit .env

docker compose up --build
```

Open the dashboard at [http://localhost:8501](http://localhost:8501).

### Using plain Docker

```bash
docker build -t 3dprintfailml:latest .
docker run -p 8501:8501 --env-file .env 3dprintfailml:latest
```

### ⚠️ Webcam access in containers

The container starts and the dashboard loads, but capturing a webcam feed has
host-specific limitations:

- **Docker Desktop on Windows/macOS cannot pass a USB webcam into a
  container.** Local-camera monitoring is only possible on a **Linux host**.
- On a Linux host, pass the camera device through and uncomment the `devices`
  block in [`docker-compose.yml`](docker-compose.yml):

  ```bash
  docker run -p 8501:8501 --env-file .env \
    --device=/dev/video0:/dev/video0 \
    3dprintfailml:latest
  ```

  > Note: the capture backend in the code uses Windows DirectShow
  > (`cv2.CAP_DSHOW`). On Linux it needs the default V4L2 backend, so the
  > capture call must be adjusted for camera access inside a Linux container.

## Training Your Own Model

The included model (`models/best.pt`) is pre-trained on 3D print failure data. To train your own:

1. Prepare your dataset in YOLO format (images + labels)
2. Update `data/data.yaml` with your class names
3. Run the training script:
   ```bash
   python scripts/train.py
   ```

### Export for Edge Devices (Raspberry Pi)

```bash
python scripts/convert.py
```

This exports the model to NCNN format for efficient inference on ARM devices.

## 📁 Project Structure

```
3DPrintFailML/
├── app.py              # Streamlit web dashboard
├── main.py             # CLI monitoring script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image for the dashboard
├── docker-compose.yml  # Compose config (port + env + webcam passthrough)
├── .dockerignore       # Build context excludes
├── .env.example        # Environment variables template
├── models/
│   └── best.pt         # Trained YOLO model
├── scripts/
│   ├── train.py        # Model training script
│   └── convert.py      # NCNN export script
└── data/
    ├── data.yaml       # Dataset configuration
    ├── train/          # Training images & labels
    ├── valid/          # Validation images & labels
    └── test/           # Test images & labels
```

## Detected Classes

| Class       | Description                             |
| ----------- | --------------------------------------- |
| `spaghetti` | Tangled filament mess, failed extrusion |
| `stringing` | Thin strings between print parts        |
| `warping`   | Print lifting or curling from bed       |

## Acknowledgments

- Dataset: [3D Print Failure Detection](https://universe.roboflow.com/purvi-rathore-5amqh/3d-print-failure-detection-efvsh/dataset/4) on Roboflow (CC BY 4.0)
- Built with [Ultralytics YOLOv8](https://ultralytics.com/)

## 📄License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The training dataset is licensed under CC BY 4.0.
