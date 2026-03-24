import cv2
import threading
import time
import requests
import os
import re
import subprocess
from datetime import datetime
from inference_sdk import InferenceHTTPClient
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from flask import Flask, send_file, Response, jsonify
from flask_cors import CORS

# ============================================================
#  CONFIG
# ============================================================
BLYNK_AUTH       = "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"
ROBOFLOW_API_KEY = "C5XOHHSHXJqy9FExFgkc"
MODEL_ID         = "nhandienrau-iajgf/1"

MAX_WATER = 10.0   # giây tưới tối đa
MAX_SPRAY = 10.0   # giây phun tối đa
SOIL_DRY  = 10     # % đất khô → bật bơm
TEMP_LOW  = 30     # °C dưới → tắt quạt
TEMP_HIGH = 34     # °C trên → bật quạt
LUX_LOW   = 500    # lux thấp → bật đèn
COOLDOWN  = 60     # giây chờ giữa 2 lần bật bơm
MODEL_FILE = "ai_model.pkl"
# ============================================================

app = Flask(__name__)
CORS(app)

os.makedirs("LogData", exist_ok=True)

# ── Roboflow client ──────────────────────────────────────────
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# ── Camera ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

frame_lock    = threading.Lock()
current_frame = None

def camera_loop():
    global current_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        time.sleep(0.033)

threading.Thread(target=camera_loop, daemon=True).start()

# ── State ────────────────────────────────────────────────────
last_result    = {}
irrigation_time = MAX_WATER
is_inferring   = False
pump_running   = False
last_pump_time = 0
last_fan       = -1
last_light     = -1
counter        = 0

# ============================================================
#  AI MODEL
# ============================================================
def train_model():
    if not os.path.exists("ai_dataset.csv"):
        return None
    df    = pd.read_csv("ai_dataset.csv")
    X     = df[["temp","hum","soil","lux","pest","wilt"]]
    y     = df[["irrigation","spray"]]
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("✅ Model trained!")
    return model

def load_model():
    return joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else train_model()

model = load_model()

def predict_ai(t, h, s, l, pest, wilt):
    global model
    if model is None:
        return MAX_WATER * wilt, MAX_SPRAY * pest
    try:
        X = pd.DataFrame([[float(t), float(h), float(s), float(l), pest, wilt]],
                         columns=["temp","hum","soil","lux","pest","wilt"])
        p = model.predict(X)[0]
        return max(0, float(p[0])), max(0, float(p[1]))
    except:
        return 0, 0

def save_dataset(t, h, s, l, pest, wilt, irr, spray):
    exists = os.path.exists("ai_dataset.csv")
    with open("ai_dataset.csv", "a") as f:
        if not exists:
            f.write("temp,hum,soil,lux,pest,wilt,irrigation,spray\n")
        f.write(f"{t},{h},{s},{l},{pest},{wilt},{irr},{spray}\n")

# ============================================================
#  BLYNK
# ============================================================
def get_blynk(pin):
    try:
        r = requests.get(
            f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH}&v{pin}",
            timeout=2)
        return r.text if r.status_code == 200 else "--"
    except:
        return "--"

def set_blynk(pin, value):
    try:
        requests.get(
            f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH}&{pin}={value}",
            timeout=2)
        print(f"  Blynk {pin} = {value}")
    except Exception as e:
        print("Blynk error:", e)

def auto_off(pin, t):
    global pump_running
    time.sleep(t)
    set_blynk(pin, 0)
    pump_running = False

# ============================================================
#  AUTO SENSOR CONTROL (chạy nền, không cần Tkinter)
# ============================================================
def sensor_loop():
    global pump_running, last_pump_time, last_fan, last_light

    while True:
        try:
            t    = get_blynk(0)
            h    = get_blynk(1)
            soil = get_blynk(2)
            lux  = get_blynk(3)
            mode = get_blynk(9)

            if mode == "1":  # AUTO
                # Bơm tưới
                soil_v = int(float(soil)) if soil not in ("--","") else 100
                if soil_v < SOIL_DRY and not pump_running:
                    if time.time() - last_pump_time > COOLDOWN:
                        pump_running   = True
                        last_pump_time = time.time()
                        set_blynk("V6", 1)
                        threading.Thread(
                            target=auto_off, args=("V6", irrigation_time), daemon=True
                        ).start()

                # Quạt
                fan = last_fan
                try:
                    tv = float(t)
                    if tv > TEMP_HIGH: fan = 1
                    elif tv < TEMP_LOW: fan = 0
                except: pass
                if fan != last_fan:
                    set_blynk("V5", fan)
                    last_fan = fan

                # Đèn sinh trưởng
                light = last_light
                hour  = datetime.now().hour
                if hour < 6 or hour > 18:
                    light = 0
                else:
                    try:
                        lv = int(float(lux))
                        if lv < LUX_LOW:           light = 1
                        elif lv > LUX_LOW + 1200:  light = 0
                    except: pass
                if light != last_light:
                    set_blynk("V4", light)
                    last_light = light
            else:
                last_light = -1

        except Exception as e:
            print("sensor_loop error:", e)

        time.sleep(2)

threading.Thread(target=sensor_loop, daemon=True).start()

# ============================================================
#  AI INFERENCE TASK
# ============================================================
COLOR_MAP = {
    "leaf":   (0, 255, 0),
    "pest":   (0, 0, 255),
    "wilt":   (0, 255, 255),
    "chit":   (255, 0, 0),
    "small":  (255, 0, 0),
    "medium": (255, 0, 0),
    "big":    (255, 0, 0),
}

def run_inference_task():
    global current_frame, irrigation_time, last_result, last_light, is_inferring, model, counter

    # Lấy frame hiện tại
    with frame_lock:
        if current_frame is None:
            print("No frame available")
            is_inferring = False
            return
        frame = current_frame.copy()

    # Bật đèn trắng để chụp, tắt đèn sinh trưởng
    set_blynk("V7", 1)
    set_blynk("V4", 0)
    last_light = -1
    time.sleep(2)

    # Lưu ảnh capture
    cv2.imwrite("capture.jpg", frame)

    # Gửi lên Roboflow
    try:
        results = client.infer("capture.jpg", model_id=MODEL_ID)
    except Exception as e:
        print("Roboflow error:", e)
        set_blynk("V7", 0)
        is_inferring = False
        return

    # Xử lý kết quả
    img = frame.copy()
    leaf_count = pest_count = wilt_count = 0
    leaf_area  = pest_area  = wilt_area  = 0

    for pred in results["predictions"]:
        x, y  = int(pred["x"]),     int(pred["y"])
        w, h  = int(pred["width"]), int(pred["height"])
        cls   = pred["class"]
        conf  = pred["confidence"]
        area  = w * h

        if   cls == "leaf": leaf_count += 1; leaf_area += area
        elif cls == "pest": pest_count += 1; pest_area += area
        elif cls == "wilt": wilt_count += 1; wilt_area += area

        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        color  = COLOR_MAP.get(cls, (255, 255, 255))
        label  = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    total_area     = leaf_area + pest_area + wilt_area
    pest_sev       = pest_area / total_area if total_area > 0 else 0
    wilt_sev       = wilt_area / total_area if total_area > 0 else 0
    stress         = pest_sev + wilt_sev

    if   stress < 0.05: status = "Healthy"
    elif stress < 0.15: status = "Stress nhẹ"
    elif stress < 0.35: status = "Stress trung bình"
    else:               status = "Stress nặng"

    # Đọc cảm biến để dự đoán
    t_v = get_blynk(0); h_v = get_blynk(1)
    s_v = get_blynk(2); l_v = get_blynk(3)
    irrigation_time, spray_time = predict_ai(t_v, h_v, s_v, l_v, pest_sev, wilt_sev)

    # Lưu dataset & retrain
    save_dataset(t_v, h_v, s_v, l_v, pest_sev, wilt_sev,
                 MAX_WATER * wilt_sev, MAX_SPRAY * pest_sev)
    counter += 1
    if counter >= 10:
        model   = train_model()
        counter = 0

    # Vẽ overlay thông tin lên ảnh
    current_time = time.strftime("%H:%M:%S - %d/%m/%Y")
    overlay  = img.copy()
    img_out  = img.copy()
    cv2.rectangle(overlay, (0, 0), (310, 230), (30, 30, 30), -1)
    img_out  = cv2.addWeighted(overlay, 0.55, img_out, 0.45, 0)

    lines = [
        f"Time:   {current_time}",
        f"Leaf:   {leaf_count + pest_count + wilt_count}",
        f"Pest:   {pest_count}  ({pest_sev:.3f})",
        f"Wilt:   {wilt_count}  ({wilt_sev:.3f})",
        f"Stress: {stress:.3f}",
        f"Status: {status}",
        f"Irrig:  {irrigation_time:.1f}s",
        f"Spray:  {spray_time:.1f}s",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img_out, line, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # Lưu ảnh
    filename = time.strftime("LogData/%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(filename, img_out)
    cv2.imwrite("capture.jpg", img_out)
    print(f"✅ Saved: {filename}")

    # Tắt đèn trắng
    set_blynk("V7", 0)

    # Phun thuốc nếu cần
    if spray_time > 0.5:
        set_blynk("V8", 1)
        threading.Thread(target=auto_off, args=("V8", spray_time), daemon=True).start()

    # Cập nhật kết quả trả về cho index.html
    last_result = {
        "time":         current_time,
        "leaf_count":   leaf_count + pest_count + wilt_count,
        "pest_count":   pest_count,
        "wilt_count":   wilt_count,
        "pest_severity": round(pest_sev, 3),
        "wilt_severity": round(wilt_sev, 3),
        "stress_index":  round(stress, 3),
        "status":        status,
        "irrigation":    round(irrigation_time, 2),
        "spray":         round(spray_time, 2),
    }

    is_inferring = False

# ============================================================
#  FLASK ROUTES
# ============================================================

def gen_frames():
    """MJPEG live stream"""
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            frame = current_frame.copy()
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    """Live camera stream cho index.html"""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    """index.html bấm nút → chụp ảnh + chạy AI"""
    global is_inferring
    if is_inferring:
        return jsonify({"status": "busy", "message": "Đang phân tích, vui lòng chờ..."})
    is_inferring = True
    threading.Thread(target=run_inference_task, daemon=True).start()
    return jsonify({"status": "ok", "message": "Đang xử lý AI..."})

@app.route("/result")
def result():
    """Trả kết quả AI mới nhất dạng JSON"""
    return jsonify(last_result)

@app.route("/capture")
def latest_result():
    """Ảnh annotate mới nhất"""
    if os.path.exists("capture.jpg"):
        return send_file("capture.jpg", mimetype="image/jpeg")
    return "No image yet", 404

@app.route("/status")
def status_check():
    """index.html poll trạng thái inference"""
    return jsonify({"inferring": is_inferring, "has_result": bool(last_result)})

# ── Cloudflare tunnel (public URL → Blynk V12) ──────────────
def run_cloudflare():
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        print(line.strip())
        m = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)
        if m:
            url = m.group(1)
            print(f"\n🔥 PUBLIC URL: {url}\n")
            set_blynk("V12", url)

threading.Thread(target=run_cloudflare, daemon=True).start()

# ============================================================
if __name__ == "__main__":
    print("🌱 Greenhouse API server → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
