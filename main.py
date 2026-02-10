import os
import cv2
import time
from dotenv import load_dotenv
from ultralytics import YOLO
from collections import deque
from discord_webhook import DiscordWebhook, DiscordEmbed

# 1. Setup & Config
load_dotenv()
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Configuration Constants
CONFIDENCE_THRESHOLD = 0.5
BUFFER_SIZE = 15
ALERT_THRESHOLD = 0.7  # Trigger if 70% of buffer is 'Failure'
COOLDOWN_SECONDS = 120  # Prevent spam (2 minutes)


def send_discord_alert(frame, failure_rate):
    """Saves a frame and sends it to Discord with an embed."""
    if not WEBHOOK_URL:
        print("Error: No Discord Webhook URL found in .env")
        return

    img_path = "latest_failure.jpg"
    cv2.imwrite(img_path, frame)

    webhook = DiscordWebhook(
        url=WEBHOOK_URL, content="🚨 **Print Failure Possibly Detected!**"
    )

    embed = DiscordEmbed(
        title="Possible 3D Print Failure Alert",
        description=f"Confidence: {failure_rate:.1%}\nStatus: Print may be failing.",
        color="ff0000",
    )

    # Attach the snapshot
    with open(img_path, "rb") as f:
        webhook.add_file(file=f.read(), filename="snapshot.jpg")

    embed.set_image(url="attachment://snapshot.jpg")
    embed.set_timestamp()
    webhook.add_embed(embed)

    try:
        webhook.execute()
        print(f"[{time.ctime()}] Discord notification sent.")
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")


def run_monitor():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)  # 0 = Default Webcam

    detection_history = deque(maxlen=BUFFER_SIZE)
    last_alert_time = 0

    print("--- 3D Print Guardian Active ---")
    print("Press 'q' in the window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Run Inference
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Logic: Was a failure detected in THIS frame?
        failure_detected = len(results[0].boxes) > 0
        detection_history.append(failure_detected)

        # Calculate 'Stable' Failure Rate
        failure_rate = (
            sum(detection_history) / len(detection_history) if detection_history else 0
        )

        # Create Annotated Frame for Display
        annotated_frame = results[0].plot()

        # Alert Logic
        current_time = time.time()
        if failure_rate > ALERT_THRESHOLD:
            # Only alert if cooldown has passed
            if (current_time - last_alert_time) > COOLDOWN_SECONDS:
                send_discord_alert(frame, failure_rate)
                last_alert_time = current_time

            # UI Overlay for Alert
            cv2.putText(
                annotated_frame,
                "!!! FAILURE DETECTED !!!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        # Show the Feed
        cv2.imshow("3D Print Guardian", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # The 'if __name__' block is MANDATORY for Windows multiprocessing
    run_monitor()
