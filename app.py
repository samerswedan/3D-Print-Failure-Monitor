import streamlit as st
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
from collections import deque
from dotenv import load_dotenv
from discord_webhook import DiscordWebhook, DiscordEmbed

# --- 1. INITIALIZATION & CONFIG ---
st.set_page_config(page_title="Guardian AI | 3D Print Monitor", layout="wide")
load_dotenv()

# Pathing safety
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# --- 2. SIDEBAR & UX CONTROLS ---
st.sidebar.title("🛠️ System Control Panel")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.5,
    help="Minimum AI confidence to flag a failure.",
)
buffer_size = st.sidebar.number_input(
    "Consistency Buffer (Frames)",
    5,
    30,
    10,
    help="Number of frames a failure must persist to trigger an alert.",
)
enable_discord = st.sidebar.checkbox("Enable Discord Alerts", value=True)

if st.sidebar.button("🧪 Test Discord Connection"):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        test_webhook = DiscordWebhook(
            url=webhook_url, content="✅ **Guardian AI Connection Test Successful!**"
        )
        test_webhook.execute()
        st.sidebar.success("Test Sent!")
    else:
        st.sidebar.error("Webhook URL missing in .env")


# --- 3. CORE LOGIC FUNCTIONS ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at: {path}")
        st.stop()
    return YOLO(path)


def send_alert(frame, conf):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        cv2.imwrite("alert_img.jpg", frame)
        webhook = DiscordWebhook(
            url=webhook_url, content="🚨 **3D PRINT FAILURE DETECTED**"
        )
        embed = DiscordEmbed(
            title="Stability Alert",
            description=f"AI confirmed failure with {conf:.2%} confidence.",
            color="ff0000",
        )
        with open("alert_img.jpg", "rb") as f:
            webhook.add_file(file=f.read(), filename="fail.jpg")
        embed.set_image(url="attachment://fail.jpg")
        webhook.add_embed(embed)
        webhook.execute()
        os.remove("alert_img.jpg")


# --- 4. APP UI HEADER ---
st.title("3D Print Monitor")
status_container = st.empty()  # For the 🟢 Status Badge
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    st.subheader("Log")
    log_placeholder = st.empty()
    if "logs" not in st.session_state:
        st.session_state.logs = []

# --- 5. MAIN MONITORING LOOP ---
model = load_model(MODEL_PATH)
detection_buffer = deque(maxlen=buffer_size)
last_alert_time = 0

# Persistent Camera Recovery Logic
while True:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_container.error("🔴 CAMERA DISCONNECTED - Attempting to reconnect...")
        time.sleep(2)
        continue

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Trigger camera reset

        # AI Inference
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # Temporal Filtering (Consistency Check)
        is_failing = len(results[0].boxes) > 0
        detection_buffer.append(is_failing)

        # Logic: If failure is seen in > 70% of the buffer
        failure_detected = sum(detection_buffer) / buffer_size > 0.7

        # Update Status Badge
        if failure_detected:
            status_container.error("🔴 ALERT: UNSTABLE PRINT DETECTED")

            conf = results[0].boxes.conf[0].item()
            curr_time = time.time()

            # Log & Alert (60s cooldown)
            if enable_discord and (curr_time - last_alert_time > 60):
                send_alert(frame, conf)
                last_alert_time = curr_time
                timestamp = time.strftime("%H:%M:%S")
                st.session_state.logs.insert(
                    0, f"[{timestamp}] Alert Sent ({conf:.2f})"
                )
        else:
            status_container.success("🟢 SYSTEM STATUS: MONITORING")

        # Update Visuals
        frame_placeholder.image(
            annotated_frame, channels="BGR", use_container_width=True
        )
        log_placeholder.code("\n".join(st.session_state.logs[:12]))

    cap.release()
    time.sleep(1)  # Small pause before camera reset
