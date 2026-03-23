# FULL OK DONE
from datetime import datetime
import cv2
import threading
import time
from tkinter import *
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient
import requests
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import subprocess
import re
from flask import Flask, send_file

Image_path = "capture.jpg"

app = Flask(__name__)
@app.route("/image")
def get_image():
    return send_file(Image_path, mimetype='image/jpeg')

def run_flask():
    app.run(host="0.0.0.0", port=5000)

def run_cloudflare():
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line.strip())

        # tìm link public
        match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)

        if match:
            public_url = match.group(1)
            print("\n🔥 PUBLIC URL:", public_url + "/image\n")
            image_url = public_url + "/image"
            blynk_write("V12", image_url)

if not os.path.exists("LogData"):
    os.makedirs("LogData")

MAX_WATER = 10.0  # giây
MAX_SPRAY = 10.0  # giây
irrigation_time = MAX_WATER  # thời gian tưới nước
SOIL_DRY = 10       # < 20% → khô
TEMP_LOW = 30       # < 30°C → tắt quạt
TEMP_HIGH = 34      # > 34°C → bật quạt
LUX_LOW = 500

# ================= CONFIG =================
BLYNK_AUTH = "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"
ROBOFLOW_API_KEY = "C5XOHHSHXJqy9FExFgkc"
MODEL_ID = "nhandienrau-iajgf/1"
# =========================================

# Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera not found")
    exit()
frame = None
capture_flag = False
counter = 0

# Sensor data
temp = "--"
hum = "--"
soil = "--"
lux = "--"

MODEL_FILE = "ai_model.pkl"

def train_model():
    if not os.path.exists("ai_dataset.csv"):
        return None

    df = pd.read_csv("ai_dataset.csv")
    X = df[["temp", "hum", "soil", "lux", "pest", "wilt"]]
    y = df[["irrigation", "spray"]]
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    print("Model trained!")

    return model

def auto_retrain():
    global counter, model
    counter += 1

    if counter >= 10:  # mỗi 10 lần chạy sẽ training
        print("Retraining model...")
        model = train_model()
        counter = 0

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return train_model()

model = load_model()

# ================= TKINTER =================
root = Tk()
root.title("Greenhouse system")
root.geometry("1200x700")

lbl_image = Label(root, bg="orange")
lbl_image.place(x=50, y=100, width=700, height=450)

txt_result = Label(root, text="Kết quả phân tích:", fg="red", font=("Arial", 16))
txt_result.place(x=800, y=100)

lbl_result = Label(root, text="", fg="brown", justify=LEFT, font=("Arial", 14))
lbl_result.place(x=800, y=130)

lbl_sensor = Label(root, text="", fg="black", font=("Arial", 14))
lbl_sensor.place(x=50, y=50)

lbl_cam = Label(root)
lbl_cam.place(x=850, y=450, width=250, height=180)

lbl_time = Label(root, fg="blue", font=("Arial", 16))
lbl_time.place(x=850, y=30)


# ================= CAMERA LOOP =================
def update_camera():
    global frame
    ret, frame = cap.read()
    if ret:
        img = cv2.resize(frame, (250, 180))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        lbl_cam.imgtk = img
        lbl_cam.configure(image=img)

    root.after(30, update_camera)


# ================= INFERENCE =================
# Bounding Box
COLOR_MAP = {
    "leaf": (0, 255, 0),  # xanh lá
    "pest": (0, 0, 255),  # đỏ
    "wilt": (0, 255, 255),  # vàng
    "chit": (255, 0, 0),  # xanh dương
    "small": (255, 0, 0),
    "medium": (255, 0, 0),
    "big": (255, 0, 0),
}


def predict_ai(temp, hum, soil, lux, pest, wilt):
    global model

    if model is None:
        return MAX_WATER * wilt, MAX_SPRAY * pest

    try:
        X = pd.DataFrame([[float(temp), float(hum), float(soil), float(lux), pest, wilt]],
                         columns=["temp", "hum", "soil", "lux", "pest", "wilt"])

        pred = model.predict(X)[0]

        irrigation = max(0, float(pred[0]))
        spray = max(0, float(pred[1]))

        return irrigation, spray
    except:
        return 0, 0


def save_dataset(temp, hum, soil, lux, pest, wilt, irrigation, spray):
    file_exists = os.path.exists("ai_dataset.csv")

    with open("ai_dataset.csv", "a") as f:
        if not file_exists:
            f.write("temp,hum,soil,lux,pest,wilt,irrigation,spray\n")

        f.write(f"{temp},{hum},{soil},{lux},{pest},{wilt},{irrigation},{spray}\n")


def run_inference():
    global frame, irrigation_time, Image_path, last_light


    if frame is None:
        return

    # ================= BLYNK ON =================
    '''
    V0: Nhiet do
    V1: Do am
    V2: Do am dat
    V3: Anh sang (lux)
    V4: Đen sinh truong
    V5: Quat
    V6: Bom
    V7: Led trang
    V8: Phun thuoc
    '''
    blynk_write("V7", 1)  # Bat den led trang de chup anh
    blynk_write("V4", 0)  # Tắt đèn Tăng trưởng
    last_light = -1
    time.sleep(3)

    path = "capture.jpg"
    cv2.imwrite(path, frame)

    results = client.infer(path, model_id=MODEL_ID) # Gui anh len Roboflow de nhan ket qua du doan(class, bounding, confidence)
    img = frame.copy()

    # ================= THỐNG KÊ =================
    leaf_count = 0
    pest_count = 0
    wilt_count = 0

    leaf_area = 0
    pest_area = 0
    wilt_area = 0

    # ================= LOOP =================
    for pred in results['predictions']:
        x = int(pred['x'])
        y = int(pred['y'])
        w = int(pred['width'])
        h = int(pred['height'])
        cls = pred['class']
        conf = pred['confidence']

        area = w * h

        # Đếm & tính diện tích
        if cls == "leaf":
            leaf_count += 1
            leaf_area += area
        elif cls == "pest":
            pest_count += 1
            pest_area += area
        elif cls == "wilt":
            wilt_count += 1
            wilt_area += area

        # Vẽ bbox
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        color = COLOR_MAP.get(cls, (255, 255, 255))  # mặc định trắng nếu không có
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ================= TÍNH TOÁN =================
    total_leaf = leaf_count + pest_count + wilt_count
    total_area = leaf_area + pest_area + wilt_area

    if total_area > 0:
        pest_severity = pest_area / total_area
        wilt_severity = wilt_area / total_area
    else:
        pest_severity = 0
        wilt_severity = 0

    stress_index = pest_severity + wilt_severity

    # ================= STATUS =================
    if stress_index < 0.05:
        status = "Healthy"
    elif stress_index < 0.15:
        status = "Stress nhẹ"
    elif stress_index < 0.35:
        status = "Stress trung bình"
    else:
        status = "Stress nặng"

    real_irrigation = MAX_WATER * wilt_severity
    real_spray = MAX_SPRAY * pest_severity

    irrigation_time, spray_time = predict_ai(temp, hum, soil, lux, pest_severity, wilt_severity)

    # ================= LƯU DATA =================
    save_dataset(temp, hum, soil, lux, pest_severity, wilt_severity, real_irrigation, real_spray)
    auto_retrain()

    # ================= HIỂN THỊ TEXT =================
    current_time = time.strftime("%H:%M:%S - %d/%m/%Y")
    text_out = (
        f"Time: {current_time}\n"
        f"Leaf detected: {total_leaf}\n"
        f"Pest detected: {pest_count}\n"
        f"Wilt detected: {wilt_count}\n"
        f"Pest severity: {pest_severity:.3f}\n"
        f"Wilt severity: {wilt_severity:.3f}\n"
        f"Plant stress index: {stress_index:.3f}\n"
        f"Status: {status}\n"
        f"Irrigation: {irrigation_time:.2f} s\n"
        f"Spray time: {spray_time:.2f} s"
    )

    # ================= HIỂN THỊ ẢNH =================
    overlay = img.copy()
    image = img.copy()
    img = cv2.resize(img, (700, 450))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(img))

    lbl_image.imgtk = img
    lbl_image.configure(image=img)
    lbl_result.config(text=text_out)

    # //////////////////////////////////////////////////////////

    # ================= TIME & SENSOR =================


    sensor_text = (
        f"Temp: {temp} *C\n"
        f"Hum: {hum} %\n"
        f"Soil: {soil} %\n"
        f"Lux: {lux} lx"
    )

    # ================= OVERLAY TEXT =================

    # nền mờ bên trái
    cv2.rectangle(overlay, (0, 0), (300, 400), (50, 50, 50), -1)
    alpha = 0.5
    img_overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    img = img_overlay

    y0 = 20
    dy = 20
    info_lines = [
        f"Time: {current_time}",
        f"Leaf detected: {total_leaf}",
        f"Pest detected: {pest_count}",
        f"Wilt detected: {wilt_count}",
        f"Pest severity: {pest_severity:.3f}",
        f"Wilt severity: {wilt_severity:.3f}",
        f"Stress index: {stress_index:.3f}",
        f"Status: {status}",
        f"Irrigation: {irrigation_time:.2f} s",
        f"Spray: {spray_time:.2f} s"
    ]

    for i, line in enumerate(info_lines):
        y = y0 + i * dy
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # SENSOR
    for i, line in enumerate(sensor_text.split("\n")):
        y = y0 + (len(info_lines) + 1 + i) * dy
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ================= LƯU ẢNH =================
    filename = time.strftime("LogData/%Y%m%d_%H%M%S.jpg")
    Image_path = filename
    cv2.imwrite(filename, img)
    print("Saved:", filename)

    # ================= TẮT BLYNK =================
    blynk_write("V7", 0)

    # if irrigation_time > 1:
    #     blynk_write("V6", 1)
    #     threading.Thread(target=auto_off, args=("V6", irrigation_time)).start()

    if spray_time > 0.5: # nhận diện ảnh để phun thuốc, tưới cây phụ thuộc vào cảm biến
        blynk_write("V8", 1)
        threading.Thread(target=auto_off, args=("V8", spray_time)).start()

    root.after(24 * 60 * 60 * 1000, run_inference)


def auto_off(pin, t):
    global pump_running
    time.sleep(t)
    blynk_write(pin, 0)
    pump_running = False


# ================= KEY EVENT =================
def on_space(event):
    threading.Thread(target=run_inference).start()


root.bind("<space>", on_space)


# ================= BLYNK =================
def get_blynk_value(pin):
    try:
        url = f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH}&v{pin}"
        response = requests.get(url, timeout=2)

        if response.status_code == 200:
            return response.text
        else:
            return "--"
    except:
        return "--"

def blynk_write(pin, value):
    try:
        url = f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH}&{pin}={value}"
        requests.get(url, timeout=2)
        print(f"Blynk {pin} = {value}")
    except Exception as e:
        print("Blynk error:", e)


# ================= UPDATE SENSOR =================
pump_running = False
last_pump_time = 0
COOLDOWN = 60  # 60s
last_fan = -1
last_light = -1
def update_sensor():
    global temp, hum, soil, lux, irrigation_time, pump_running, last_pump_time
    global last_fan, last_light, auto_mode

    temp = get_blynk_value(0)
    hum = get_blynk_value(1)
    soil = get_blynk_value(2)
    lux = get_blynk_value(3)

    txt = f"Temp: {temp} °C\tHumidity: {hum} %\tSoil: {soil} %\tLux: {lux} lx"
    lbl_sensor.config(text=txt)

    if auto_mode == 1:
        soil = int(soil) if str(soil).isdigit() else 0
        if soil < SOIL_DRY and not pump_running:
            if time.time() - last_pump_time > COOLDOWN:  # Sau COOLDOWN mới bật lại bơm tránh bật tắt liên tục
                pump_running = True
                last_pump_time = time.time()
                blynk_write("V6", 1)  # bật bơm tưới nhỏ giọt
                threading.Thread(target=auto_off, args=("V6", irrigation_time),daemon=True).start()

        fan = last_fan
        temp = float(temp) if str(temp).isdigit() else 0
        if float(temp) > TEMP_HIGH:
            fan = 1
        elif float(temp) < TEMP_LOW:
            fan = 0
        if fan != last_fan:
            blynk_write("V5", fan)
            last_fan = fan

        light = last_light
        hour = datetime.now().hour
        if hour < 6 or hour > 18:
            light = 0
        else:
            lux = int(lux) if str(lux).isdigit() else 0
            if lux < LUX_LOW:
                light = 1
            elif lux > LUX_LOW + 1200:
                light = 0
        if light != last_light:
            blynk_write("V4", light)  # đèn sinh trưởng: ban đêm nghỉ, ban ngày sáng yếu mới bật
            last_light = light
    else:
        last_light = -1

    root.after(2000, update_sensor)  # mỗi 2s gọi lại


is_running = False
auto_mode = 0


def check_trigger():
    global is_running, auto_mode

    try:
        auto_mode = int(get_blynk_value(9) or 0)
        trigger = int(get_blynk_value(10) or 0)

        if auto_mode == 1 and trigger == 1 and not is_running:
            print("Auto mode + Trigger!")

            is_running = True

            def task():
                global is_running
                run_inference()
                is_running = False

            threading.Thread(target=task).start()
            # reset trigger
            requests.get(f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH}&v10=0")

    except Exception as e:
        print("Lỗi:", e)

    root.after(1000, check_trigger)


# ================= TIME =================
def update_time():
    now = time.strftime("%H:%M:%S - %d/%m/%Y")
    lbl_time.config(text=now)
    root.after(1000, update_time)


threading.Thread(target=run_flask, daemon=True).start()
time.sleep(2)  # đợi flask chạy
threading.Thread(target=run_cloudflare).start()

# ================= START =================
update_camera()
update_sensor()
update_time()
check_trigger()
root.mainloop()


