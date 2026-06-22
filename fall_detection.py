import os

os.environ[
    "OPENCV_FFMPEG_CAPTURE_OPTIONS"
] = "rtsp_transport;tcp|stimeout;5000000"

import cv2
import numpy as np
import time
import math
import threading
from collections import deque
import tensorflow as tf


import paho.mqtt.client as mqtt
import base64
import datetime
import subprocess
import requests
import tempfile

from datetime import datetime
import json
import cloudinary
import cloudinary.uploader
from twilio.rest import Client


# ==================================================
# RTSP CAMERAS
# ==================================================

RTSP_URLS = [

    "rtsp://pulkitgarg:Allenhouse@123@192.168.1.57:554/stream1",

    "rtsp://pulkitgarg:Allenhouse@123@192.168.1.60:554/stream2",

    # Add more cameras here
]

MODEL_PATH = "movenet_singlepose_thunder.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])


camera_frames = [None] * len(RTSP_URLS)

camera_online = [False] * len(RTSP_URLS)

camera_last_frame_time = [
    0
] * len(RTSP_URLS)


frame_locks = [
    threading.Lock()
    for _ in RTSP_URLS
]



client = mqtt.Client()
try:
    client.connect("broker.hivemq.com", 1883, 60)
    client.loop_start()
except Exception as e:
    print("MQTT connection failed:", e)
    


# Cloudinary
cloudinary.config(
    cloud_name="dl2fhwcl5",
    api_key="971978761231223",
    api_secret="G71zlwDG-zH65inED7kJn55px1M"
)

# Twilio
# ACCOUNT_SID = "AC77887f9c53cbc897caaa895720a3d88e"
# ACCOUNT_SID = "ACa26fd4ddf397d09867a3c7fc6c812b06"
# ACCOUNT_SID = "AC6bf5e6aa371f2d6cd0ac0945c44973bb"
ACCOUNT_SID = "***"



# AUTH_TOKEN = "5cdf7097f3879db06e14bf06441b3a1d"
# AUTH_TOKEN = "cddc3f97b5c10286942da0b3fa22dd09"
# AUTH_TOKEN = "6954ccdfc3120e1d885dcbe5fd965dcf"
AUTH_TOKEN = "***"



twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

TWILIO_WHATSAPP = "whatsapp:+14155238886"  # Sandbox

def upload_frame_to_cloudinary(frame):

    _, buffer = cv2.imencode(".jpg", frame)

    result = cloudinary.uploader.upload(
        buffer.tobytes(),
        resource_type="image",
        format="jpg",
        quality="auto:good"
    )

    url = result["secure_url"]

    # wait until URL is reachable (IMPORTANT FIX)
    for _ in range(5):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return url
        except:
            pass

        time.sleep(1)

    return url

def send_whatsapp_alert(image_url, cam_id):

    time.sleep(2)
    caregivers = [
        "whatsapp:+918953193403",
        "whatsapp:+916382659267",
    ]

    try:
        for number in caregivers:

            message = twilio_client.messages.create(
                from_=TWILIO_WHATSAPP,
                to=number,
                body=(
                    f"⚠ ELDER FALL DETECTED\n"
                    f"Camera: {cam_id + 1}\n"
                    f"Time: {time.strftime('%H:%M:%S')}"
                ),
                media_url=[image_url]
            )

            print("WhatsApp sent:", message.sid)
    
    except Exception as e:
        print("Twilio failed:", e)


def trigger_fall_alert(cam_index):

    def worker():

        try:
            print(f"[ALERT] Fall detected CAM {cam_index+1}")

            # Wait 2 seconds first
            time.sleep(1.5)

            # Get latest frame AFTER 2 sec
            with frame_locks[cam_index]:
                latest_frame = (
                    camera_frames[cam_index].copy()
                    if camera_frames[cam_index] is not None
                    else None
                )

            if latest_frame is None:
                print("No frame available")
                return

            # Upload image
            image_url = upload_frame_to_cloudinary(
                latest_frame
            )

            threading.Thread(
                target=send_mqtt_alert,
                args=(cam_index, "FALL_DETECTED", latest_frame),
                daemon=True
            ).start()

            threading.Thread(
                target=ring_alarm,
                daemon=True
            ).start()

            threading.Thread(
                target=send_whatsapp_alert,
                args=(image_url, cam_index),
                daemon=True
            ).start()

        except Exception as e:
            print("[ALERT ERROR]", e)

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


def camera_worker(cam_index):

    while True:

        cap = None

        try:

            print(
                f"[CAM {cam_index+1}] Connecting..."
            )

            cap = cv2.VideoCapture(
                RTSP_URLS[cam_index]
            )

            if not cap.isOpened():

                print(
                    f"[CAM {cam_index+1}] Offline"
                )

                time.sleep(5)
                continue

            print(
                f"[CAM {cam_index+1}] Connected"
            )

            camera_online[cam_index] = True

            while True:

                ret, frame = cap.read()

                if not ret:

                    print(
                        f"[CAM {cam_index+1}] Lost"
                    )

                    camera_online[cam_index] = False

                    with frame_locks[cam_index]:
                        camera_frames[cam_index] = None

                    break

                with frame_locks[cam_index]:
                    camera_frames[cam_index] = frame.copy()

                camera_last_frame_time[cam_index] = time.time()
                camera_online[cam_index] = True

        except Exception as e:

            print(
                f"[CAM {cam_index+1}] {e}"
            )

        camera_online[cam_index] = False

        with frame_locks[cam_index]:
            camera_frames[cam_index] = None

        try:
            if cap is not None:
                cap.release()
        except:
            pass

        camera_online[cam_index] = False

        with frame_locks[cam_index]:
            camera_frames[cam_index] = None

        time.sleep(5)



# ==========================================
# WELLNESS CHECK
# ==========================================

last_movement_time = time.time()
wellness_active = False
wellness_attempts = 0
wellness_reference_time = 0
last_prompt_time = time.time()
last_activity_display = datetime.now().strftime("%H:%M:%S")
last_wellness_check_time = time.time()

# ==========================
# KEYPOINT CONNECTIONS
# ==========================

EDGES = [
    (0,1),(0,2),
    (1,3),(2,4),
    (0,5),(0,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,6),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

def get_timeout():

    hour = datetime.now().hour

    if hour >= 22 or hour < 6:
        return 30 * 60

    return 1 * 60


# ==========================
# POSE ESTIMATION
# ==========================

def detect_pose(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_img = cv2.resize(img, (192, 192))

    input_img = np.expand_dims(input_img, axis=0)
    input_img = input_img.astype(np.uint8)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_img
    )

    interpreter.invoke()

    keypoints = interpreter.get_tensor(
        output_details[0]['index']
    )

    return keypoints[0][0]

# ==========================
# POSTURE CLASSIFICATION
# ==========================

def classify_posture(keypoints, threshold=0.2):

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    LEFT_HIP = 11
    RIGHT_HIP = 12

    LEFT_KNEE = 13
    RIGHT_KNEE = 14

    required = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE
    ]

    for idx in required:
        if keypoints[idx][2] < threshold:
            return "UNKNOWN"

    # Centers
    shoulder_x = (
        keypoints[LEFT_SHOULDER][1] +
        keypoints[RIGHT_SHOULDER][1]
    ) / 2

    shoulder_y = (
        keypoints[LEFT_SHOULDER][0] +
        keypoints[RIGHT_SHOULDER][0]
    ) / 2

    hip_x = (
        keypoints[LEFT_HIP][1] +
        keypoints[RIGHT_HIP][1]
    ) / 2

    hip_y = (
        keypoints[LEFT_HIP][0] +
        keypoints[RIGHT_HIP][0]
    ) / 2

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    torso_angle = abs(
        math.degrees(
            math.atan2(dy, dx)
        )
    )

    knee_y = (
        keypoints[LEFT_KNEE][0] +
        keypoints[RIGHT_KNEE][0]
    ) / 2

    hip_to_knee = abs(knee_y - hip_y)

    # Classification

    if torso_angle < 35:
        posture = "LYING"

    elif hip_to_knee < 0.20:
        posture = "SITTING"

    else:
        posture = "STANDING"

    return posture


# ==========================
# DRAW SKELETON
# ==========================

def draw_pose(frame, keypoints, threshold=0.3):

    h, w, _ = frame.shape

    points = []

    for kp in keypoints:

        y, x, conf = kp

        px = int(x * w)
        py = int(y * h)

        points.append((px, py, conf))

        if conf > threshold:
            cv2.circle(
                frame,
                (px, py),
                5,
                (0,255,0),
                -1
            )

    for p1, p2 in EDGES:

        if (
            points[p1][2] > threshold and
            points[p2][2] > threshold
        ):

            cv2.line(
                frame,
                (points[p1][0], points[p1][1]),
                (points[p2][0], points[p2][1]),
                (255,0,0),
                2
            )




def send_mqtt_alert(cam_id, event, frame=None):

    payload = {
        "camera": cam_id+1,
        "event": event,
        "time": time.time()
    }

    if frame is not None:

        # Reduce image size
        small = cv2.resize(frame, (320, 180))

        # Compress JPEG
        _, buffer = cv2.imencode(
            ".jpg",
            small,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )

        image_b64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        payload["image"] = image_b64

    client.publish(
        "eldercare/alert",
        json.dumps(payload)
    )

    print(
        f"MQTT SENT: CAM {cam_id+1} {event}"
    )


def reconnect_camera_async(cam_index):

    global caps

    state = camera_states[cam_index]

    if state["reconnecting"]:
        return

    state["reconnecting"] = True

    def worker():

        try:

            print(f"[CAM {cam_index+1}] Reconnecting...")

            cap = cv2.VideoCapture(
                RTSP_URLS[cam_index],
                cv2.CAP_FFMPEG
            )

            if cap.isOpened():

                print(
                    f"[CAM {cam_index+1}] Reconnected"
                )

                caps[cam_index] = cap

            else:

                caps[cam_index] = None

        except Exception as e:

            print(
                f"[CAM {cam_index+1}] Reconnect failed: {e}"
            )

        finally:

            state["reconnecting"] = False

    threading.Thread(
        target=worker,
        daemon=True
    ).start()
    

# ==================================================
# OPEN CAMERAS
# ==================================================

for i in range(len(RTSP_URLS)):

    threading.Thread(
        target=camera_worker,
        args=(i,),
        daemon=True
    ).start()

# ==================================================
# CAMERA STATES
# ==================================================

camera_states = []

for _ in RTSP_URLS:

    camera_states.append({

        "prev_hip_y": None,

        "rapid_drop": False,

        "lying_start": None,

        "fall_detected": False,

        "posture_history": deque(maxlen=10),

        "hip_history": deque(maxlen=5),

        "sit_triggered": False,

        "last_seen": "Never",

        "prev_keypoints": None,

        "last_reconnect_attempt": 0,

        "reconnecting": False,

        "last_movement": "Never",

        "last_visible": "Never",

        "last_visible_timestamp": 0,
        "last_movement_timestamp": time.time(),

        "last_visible_time": time.time(),

        "head_drop_detected": False,
        
        "last_head_alert": 0,

        "fall_buffer": deque(maxlen=5),
        "fall_score": 0,
        "person_detected": False,
        "last_person_seen": 0,

    })

# ==================================================
# ALARM
# ==================================================

alarm_running = False
prev_keypoints = None


def ring_alarm():

    global alarm_running

    if alarm_running:
        return

    alarm_running = True

    try:

        proc = subprocess.Popen(
            [
                "cvlc",
                "--play-and-exit",
                "Alarm01.wav"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        proc.wait()

    finally:
        alarm_running = False

prev_time = time.time()

audio_playing = False

def calculate_angle(a, b, c):
    """
    angle ABC
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    angle = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    return angle

def play_move_yourself_audio():

    global audio_playing

    if audio_playing:
        return

    audio_playing = True

    try:
        proc = subprocess.Popen(
            [
                "cvlc",
                "--play-and-exit",
                "move.mp3"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        proc.wait()

    finally:
        audio_playing = False

def start_wellness_check():

    global wellness_active
    global wellness_attempts
    global last_prompt_time
    global wellness_reference_time
    global last_wellness_check_time
    last_wellness_check_time = time.time()

    
    latest_visible = max(
        state["last_visible_timestamp"]
        for state in camera_states
    )

    any_person_present = any(
        state["person_detected"]
        for state in camera_states
    )

    if not any_person_present:
        return
    
    wellness_active = True
    wellness_attempts = 0
    last_prompt_time = 0
    wellness_reference_time = latest_visible

    print(
        "[CHECK] Wellness check started"
    )

def process_wellness_check():

    global wellness_active
    global wellness_attempts
    global last_prompt_time
    global wellness_reference_time

    if not wellness_active:
        return

    latest_movement = max(
        state["last_movement_timestamp"]
        for state in camera_states
    )

    # Person became visible again
    if latest_movement > wellness_reference_time:

        print(
            "[CHECK] Person visible again. Wellness cancelled."
        )

        wellness_active = False
        wellness_attempts = 0

        return

    now = time.time()

    # Prompt every 20 sec
    if (
        wellness_attempts < 2
        and
        now - last_prompt_time > 60
    ):

        print(
            f"[CHECK] Prompt {wellness_attempts+1}/3"
        )

        threading.Thread(
            target=play_move_yourself_audio,
            daemon=True
        ).start()

        last_prompt_time = now
        wellness_attempts += 1

    # After 3 prompts and still no visibility
    if (
        wellness_attempts >= 2
        and
        now - last_prompt_time > 60
    ):

        print("[CHECK] NO RESPONSE")

        threading.Thread(
            target=ring_alarm,
            daemon=True
        ).start()

        send_mqtt_alert(
            -1,
            "NO_RESPONSE"
        )

        blank = np.zeros(
            (360,640,3),
            dtype=np.uint8
        )

        cv2.putText(
            blank,  
            "NO RESPONSE",
            (120,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        image_url = upload_frame_to_cloudinary(blank)

        threading.Thread(
            target=send_whatsapp_alert,
            args=(image_url, -1),
            daemon=True
        ).start()

        wellness_active = False

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    frames = []

    current_time = time.time()

    for cam_index in range(len(RTSP_URLS)):

        state = camera_states[cam_index]


        with frame_locks[cam_index]:

            latest = camera_frames[cam_index]

        stale = (
            time.time()
            - camera_last_frame_time[cam_index]
        ) > 5

        if latest is None or stale:

            frame = np.zeros(
                (360,640,3),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                f"CAM {cam_index+1} OFFLINE",
                (120,180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            frames.append(frame)

            continue

        frame = latest.copy()
        
        frame = cv2.resize(frame, (640, 360))

        try:
            keypoints = detect_pose(frame)
        except Exception as e:
            print(f"Pose detection error on CAM {cam_index+1}: {e}")
            keypoints = None            
       


        # =====================================
# ACTIVITY DETECTION
# =====================================

        movement_detected = False

        if keypoints is not None:

            if state["prev_keypoints"] is not None:

                diff = np.mean(
                    np.abs(
                        keypoints[:, :2]
                        -
                        state["prev_keypoints"][:, :2]
                    )
                )

                if diff > 0.02:

                    movement_detected = True

            state["prev_keypoints"] = keypoints.copy()

        if movement_detected and state["person_detected"]:

            last_movement_time = time.time()

            state["last_movement_timestamp"] = time.time()

            state["last_movement"] = datetime.now().strftime(
                "%H:%M:%S"
            )

            latest_movement = max(
                state["last_movement_timestamp"]
                for state in camera_states
            )


            inactivity_time = (
                time.time() - last_movement_time
            )

        if keypoints is not None:

            draw_pose(frame, keypoints)            

            # =====================================
            # TEMPORAL SMOOTHING
            # =====================================

            current_posture = classify_posture(keypoints)

            state["posture_history"].append(current_posture)

            if state["posture_history"]:

                stable_posture = max(
                    set(state["posture_history"]),
                    key=state["posture_history"].count
                )

            else:

                stable_posture = "UNKNOWN"

            avg_conf = np.mean(keypoints[:, 2])

            person_present = (
                avg_conf > 0.30 and
                stable_posture != "UNKNOWN"
            )

            if person_present:

                state["person_detected"] = True
                state["last_person_seen"] = time.time()

                state["last_visible"] = datetime.now().strftime("%H:%M:%S")
                state["last_visible_timestamp"] = time.time()

            if (
                time.time() - state["last_person_seen"]
            ) > 5:

                state["person_detected"] = False

            # =====================================
            # HIP TRACKING
            # =====================================

            rapid_drop = False

            hip_y = (
                keypoints[11][0] +
                keypoints[12][0]
            ) / 2

            state["hip_history"].append(hip_y)

            smooth_hip_y = np.mean(
                state["hip_history"]
            )

            if stable_posture == "UNKNOWN":

                state["prev_hip_y"] = smooth_hip_y
                state["fall_buffer"].append(0)

            else:

                if state["prev_hip_y"] is not None:

                    vertical_drop = (
                        smooth_hip_y -
                        state["prev_hip_y"]
                    )

                    low_confidence = avg_conf < 0.35

                    if (
                        vertical_drop > 0.035 and
                        low_confidence
                    ):
                        rapid_drop = True

                        print(
                            f"[CAM {cam_index+1}] "
                            f"drop={vertical_drop:.3f} "
                            f"conf={avg_conf:.3f} "
                            f"rapid={rapid_drop}"
                        )

                state["prev_hip_y"] = smooth_hip_y

                state["fall_buffer"].append(
                    1 if rapid_drop else 0
                )
            # =====================
            # STABILITY CHECK
            # =====================
            cooldown = time.time() - state.get("last_fall_time", 0)
            fall_votes = sum(state["fall_buffer"])

            fall_confirmed = (
                sum(state["fall_buffer"]) >= 1 and
                cooldown >= 10
            )
            
            # =====================
            # TRIGGER ALERT
            # =====================
            if fall_confirmed and not state["fall_detected"]:

                state["fall_detected"] = True
                state["last_fall_time"] = time.time()

                trigger_fall_alert(cam_index)

            # =====================
            # RESET LOGIC
            # =====================
            if state["fall_detected"]:
                if time.time() - state["last_fall_time"] > 10:
                    state["fall_detected"] = False

            # =====================================
            # DISPLAY
            # =====================================

            
            cv2.putText(
                frame,
                f"POSTURE: {stable_posture}",
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )
     
            if state["fall_detected"]:

                cv2.putText(
                    frame,
                    "FALL DETECTED",
                    (20,250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )
        
        cv2.putText(
                frame,
                f"CAM {cam_index+1}",
                (20,280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2
            )

        cv2.putText(
            frame,
            f"LAST MOVEMENT: {state['last_movement']}",
            (20,310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )


        frames.append(frame)

    

    for state in camera_states:

        if (
            time.time()
            - state["last_person_seen"]
        ) > 5:

            state["person_detected"] = False



    inactivity_time = (
        time.time() - last_movement_time
    )

    WELLNESS_TIMEOUT = get_timeout()

    any_person_present = any(
        state["person_detected"]
        for state in camera_states
    )

    if (
        any_person_present
        and inactivity_time > WELLNESS_TIMEOUT
        and not wellness_active
    ):
        start_wellness_check()


    process_wellness_check()

    # =====================================
    # FPS
    # =====================================

    fps = 1 / (time.time() - prev_time)

    prev_time = time.time()

    # =====================================
    # COMBINE DISPLAY
    # =====================================

    if len(frames) == 1:

        combined = frames[0]

    elif len(frames) == 2:

        combined = np.hstack(frames)

    else:

        rows = []

        for i in range(0, len(frames), 2):

            if i + 1 < len(frames):

                row = np.hstack(
                    [frames[i], frames[i+1]]
                )

            else:

                blank = np.zeros_like(frames[i])

                row = np.hstack(
                    [frames[i], blank]
                )

            rows.append(row)

        combined = np.vstack(rows)
       

    cv2.imshow(
        "Multi Camera Fall Detection",
        combined
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==================================================
# CLEANUP
# ==================================================

cv2.destroyAllWindows()
