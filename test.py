# ======================== GREENHOUSE SMART CONTROL ========================
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

# ======================== CONFIG ========================
Image_path = "capture.jpg"
MAX_WATER = 10.0
MAX_SPRAY = 10.0
SOIL_DRY = 10
TEMP_LOW = 30
TEMP_HIGH = 34
LUX_LOW = 500
COOLDOWN = 60

BLYNK_AUTH = "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"
ROBOFLOW_API_KEY = "C5XOHHSHXJqy9FExFgkc"
MODEL_ID = "nhandienrau-iajgf/1"
MODEL_FILE = "ai_model.pkl"

# ======================== FLASK FOR IMAGE ========================
app = Flask(__name__)
if not os.path.exists("LogData"):
    os.makedirs("LogData")

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
        match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)
        if match:
            public_url = match.group(1)
            print("\n🔥 PUBLIC URL:", public_url + "/image\n")
            image_url = public_url + "/image"
            blynk_write("V12", image_url)

# ======================== ROBOTFLOW CLIENT ========================
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# ======================== CAMERA ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    exit()
frame = None

# ======================== SENSOR + DELTA ========================
temp = hum = soil = lux = "--"
last_soil = None
last_pest = None
last_wilt = None
delta_soil = delta_pest = delta_wilt = 0
pump_running = False
last_pump_time = 0
last_fan = -1
last_light = -1
auto_mode = 0

# ======================== AI MODEL ========================
def train_model():
    if not os.path.exists("ai_dataset.csv"):
        return None
    df = pd.read_csv("ai_dataset.csv")
    X = df[["temp","hum","soil","lux","pest","wilt","delta_soil","delta_pest"]]
    y = df[["irrigation","spray"]]
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X,y)
    joblib.dump(model, MODEL_FILE)
    print("Model trained!")
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_model()

model = load_model()
counter = 0
def auto_retrain():
    global counter, model
    counter += 1
    if counter >= 10:
        print("Retraining model...")
        model = train_model()
        counter = 0

def predict_ai(temp, hum, soil, lux, pest, wilt, delta_soil=0, delta_pest=0):
    global model
    if model is None:
        return MAX_WATER*wilt, MAX_SPRAY*pest
    try:
        X = pd.DataFrame([[
            float(temp), float(hum), float(soil), float(lux),
            pest, wilt, delta_soil, delta_pest
        ]], columns=["temp","hum","soil","lux","pest","wilt","delta_soil","delta_pest"])
        pred = model.predict(X)[0]
        irrigation = max(0,float(pred[0]))
        spray = max(0,float(pred[1]))
        if delta_soil < -2:
            irrigation *= 1.2
        irrigation = min(irrigation, MAX_WATER)
        if delta_pest > 0.05:
            spray *= 1.3
        spray = min(spray, MAX_SPRAY)
        return irrigation, spray
    except Exception as e:
        print("Predict AI error:", e)
        return 0,0

# ======================== TKINTER GUI ========================
root = Tk()
root.title("Greenhouse system")
root.geometry("1200x700")
lbl_image = Label(root, bg="orange")
lbl_image.place(x=50, y=100, width=700, height=450)
lbl_result = Label(root, text="", fg="brown", justify=LEFT, font=("Arial",14))
lbl_result.place(x=800, y=130)
lbl_sensor = Label(root, text="", fg="black", font=("Arial",14))
lbl_sensor.place(x=50,y=50)
lbl_cam = Label(root)
lbl_cam.place(x=850, y=450, width=250, height=180)
lbl_time = Label(root, fg="blue", font=("Arial",16))
lbl_time.place(x=850,y=30)

COLOR_MAP = {"leaf":(0,255,0),"pest":(0,0,255),"wilt":(0,255,255)}

def update_camera():
    global frame
    ret, frame = cap.read()
    if ret:
        img = cv2.resize(frame,(250,180))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        lbl_cam.imgtk = img
        lbl_cam.configure(image=img)
    root.after(30, update_camera)

# ======================== SENSOR UPDATE ========================
def update_sensor():
    global temp, hum, soil, lux, pump_running, last_pump_time, last_soil, delta_soil
    global last_fan, last_light, auto_mode
    temp = get_blynk_value(0)
    hum = get_blynk_value(1)
    soil_raw = get_blynk_value(2)
    lux = get_blynk_value(3)
    soil = int(soil_raw) if str(soil_raw).isdigit() else 0
    if last_soil is not None:
        delta_soil = soil - last_soil
    last_soil = soil
    lbl_sensor.config(text=f"Temp:{temp}°C  Hum:{hum}%  Soil:{soil}%  Lux:{lux} lx")
    if auto_mode==1:
        if soil<SOIL_DRY and not pump_running:
            if time.time()-last_pump_time>COOLDOWN:
                pump_running=True
                last_pump_time=time.time()
                blynk_write("V6",1)
                threading.Thread(target=auto_off,args=("V6",MAX_WATER),daemon=True).start()
    root.after(2000, update_sensor)

# ======================== AI INFERENCE ========================
def run_inference():
    global frame, irrigation_time, Image_path, last_pest, delta_pest, last_wilt, delta_wilt
    if frame is None: return
    path="capture.jpg"
    cv2.imwrite(path,frame)
    results = client.infer(path, model_id=MODEL_ID)
    leaf_count=pest_count=wilt_count=0
    leaf_area=pest_area=wilt_area=0
    for pred in results['predictions']:
        cls=pred['class']; w=int(pred['width']); h=int(pred['height'])
        area=w*h
        if cls=="leaf": leaf_count+=1; leaf_area+=area
        elif cls=="pest": pest_count+=1; pest_area+=area
        elif cls=="wilt": wilt_count+=1; wilt_area+=area
    total_area=leaf_area+pest_area+wilt_area
    pest_severity=pest_area/total_area if total_area>0 else 0
    wilt_severity=wilt_area/total_area if total_area>0 else 0
    global delta_pest, delta_wilt
    if last_pest is not None: delta_pest=pest_severity-last_pest
    last_pest=pest_severity
    if last_wilt is not None: delta_wilt=wilt_severity-last_wilt
    last_wilt=wilt_severity
    irrigation_time,spray_time=predict_ai(temp,hum,soil,lux,pest_severity,wilt_severity,delta_soil,delta_pest)
    if irrigation_time>0.5:
        if not pump_running:
            blynk_write("V6",1)
            threading.Thread(target=auto_off,args=("V6",irrigation_time),daemon=True).start()
    if spray_time>0.5:
        blynk_write("V8",1)
        threading.Thread(target=auto_off,args=("V8",spray_time),daemon=True).start()
    img=frame.copy()
    filename=time.strftime("LogData/%Y%m%d_%H%M%S.jpg")
    Image_path=filename
    cv2.imwrite(filename,img)
    save_dataset(temp,hum,soil,lux,pest_severity,wilt_severity,irrigation_time,spray_time)
    auto_retrain()

# ======================== HELPERS ========================
def auto_off(pin,t):
    global pump_running
    time.sleep(t)
    blynk_write(pin,0)
    pump_running=False

def get_blynk_value(pin):
    try:
        url=f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH}&v{pin}"
        r=requests.get(url,timeout=2)
        return r.text
    except: return "--"

def blynk_write(pin,value):
    try:
        url=f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH}&{pin}={value}"
        requests.get(url,timeout=2)
    except: pass

def save_dataset(temp,hum,soil,lux,pest,wilt,irrigation,spray):
    file_exists=os.path.exists("ai_dataset.csv")
    with open("ai_dataset.csv","a") as f:
        if not file_exists: f.write("temp,hum,soil,lux,pest,wilt,delta_soil,delta_pest,irrigation,spray\n")
        f.write(f"{temp},{hum},{soil},{lux},{pest},{wilt},{delta_soil},{delta_pest},{irrigation},{spray}\n")

# ======================== TIME ========================
def update_time():
    lbl_time.config(text=time.strftime("%H:%M:%S - %d/%m/%Y"))
    root.after(1000,update_time)

# ======================== TRIGGER ========================
def check_trigger():
    global auto_mode
    try:
        auto_mode=int(get_blynk_value(9) or 0)
        trigger=int(get_blynk_value(10) or 0)
        if auto_mode==1 and trigger==1:
            threading.Thread(target=run_inference).start()
            requests.get(f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH}&v10=0")
    except: pass
    root.after(1000,check_trigger)

# ======================== START ========================
threading.Thread(target=run_flask,daemon=True).start()
time.sleep(2)
threading.Thread(target=run_cloudflare).start()
update_camera()
update_sensor()
update_time()
check_trigger()
root.mainloop()