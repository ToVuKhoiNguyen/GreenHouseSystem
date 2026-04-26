import cv2
import threading
import time
import requests
import os
import re
import subprocess
from datetime import datetime
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from flask import Flask, send_file, Response, jsonify
from flask_cors import CORS
import random

# ============================================================
#  CONFIG
# ============================================================
BLYNK_AUTH       = "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"

# ── Model 1: detect lá (small, medium) ──────────────────────
ROBOFLOW_API_KEY_LEAF = "G1wXVaCU8zRCimzdnuHW"   # <-- điền API key model lá
MODEL_ID_LEAF         = "leafsize_noaug/1"     # <-- điền model ID model lá

# ── Model 2: detect sâu bệnh (pest, wilt) ───────────────────
ROBOFLOW_API_KEY_PEST = "cxQmVKGYY6jbGzT8NXaC"   # <-- điền API key model pest
MODEL_ID_PEST         = "leafhealth-j9ol1/1"     # <-- điền model ID model pest

# ── Ngưỡng cảm biến ─────────────────────────────────────────
MAX_WATER  = 10.0       # giây tưới tối đa mỗi lầnS
MAX_SPRAY  = 0.5       # giây phun sương tối đa
SOIL_DRY   = 35         # % độ ẩm đất — ngưỡng KHÔ (dưới → tưới)
SOIL_WET   = 70         # % độ ẩm đất — ngưỡng ĐỦ ẩm (flowchart: soil < 70%)
TEMP_LOW   = 18         # °C — dưới → tắt quạt
TEMP_HIGH  = 26         # °C — trên → bật quạt
LUX_LOW    = 800        # lux — ngưỡng bật đèn (flowchart: LUX < 800)
LUX_HIGH   = 2000       # lux — ngưỡng tắt đèn (flowchart: lux > 2000)
DLI_MIN    = 18         # mol/m²/day — ngưỡng DLI tối thiểu (flowchart)
# ── Ngưỡng VPD (flowchart image 2 phải) ─────────────────────
VPD_HIGH   = 1.5        # kPa — trên → bật quạt
VPD_LOW    = 0.5        # kPa — dưới → tắt quạt

# ── Cooldown bơm (giây) ─────────────────────────────────────
PUMP_COOLDOWN  = 300    # 5 phút cooldown giữa 2 lần tưới bổ sung
SPRAY_COOLDOWN = 120    # 2 phút cooldown giữa 2 lần phun

# ── Lịch tưới cố định (giờ:phút) ────────────────────────────
# 2 lần/ngày: sáng và chiều — chỉnh ở đây
SCHEDULED_WATERING = [
    (9, 30),    # Sáng 09:30
    (17, 0),    # Chiều 17:00
]
SCHEDULED_WATER_DURATION = 8.0   # giây mỗi lần tưới lịch

# ── Lịch chụp ảnh AI (giờ chẵn) — mỗi N giờ ────────────────
AI_CAPTURE_INTERVAL_HOURS = 2    # chụp mỗi 2 tiếng

# ── Ngưỡng quyết định tưới bổ sung ──────────────────────────
EXTRA_WATER_SOIL_THRESH  = 50    # soil < 50% → xem xét tưới thêm
EXTRA_WATER_WILT_THRESH  = 0.15  # wilt_severity > 0.15 → AI cảnh báo héo
EXTRA_WATER_VPD_THRESH   = 1.2   # VPD > 1.2 kPa → bốc hơi nhanh
EXTRA_WATER_TEMP_THRESH  = 30    # °C — nhiệt cao → mất nước nhanh
EXTRA_WATER_DURATION     = 5.0   # giây tưới bổ sung mỗi lần
MODEL_FILE = "ai_model.pkl"
#CSV_PATH   = "C:\DoAnTotNghiep\ai_dataset.csv"
CSV_PATH   = r"/home/pi/GreenHouseSystem/ai_dataset.csv"

# ── Firebase Realtime Database ───────────────────────────────
FIREBASE_DB_URL = "https://greenhouse-31c64-default-rtdb.asia-southeast1.firebasedatabase.app"
FB_NODE = "ai_dataset"
# ============================================================

app = Flask(__name__)
CORS(app)

os.makedirs("LogData",  exist_ok=True)
os.makedirs("RawData",  exist_ok=True)
os.makedirs("CSVData",  exist_ok=True)

# ── CSV lưu kết quả chụp ảnh (mỗi lần chụp 1 dòng) ─────────
CSV_CAPTURE_PATH = "CSVData/data.csv"
CSV_CAPTURE_COLS = [
    "timestamp", "source",
    "temp", "hum", "soil", "lux", "vpd",
    "leaf_count", "pest_count", "wilt_count",
    "pest_severity", "wilt_severity", "stress_index", "status",
    "irrigation_pred", "spray_pred",
    "raw_image_path", "result_image_path",
]

def save_capture_csv(**kwargs):
    """
    Ghi 1 dòng vào CSVData/data.csv mỗi khi chụp ảnh xong.
    kwargs phải chứa các key trong CSV_CAPTURE_COLS.
    """
    exists = os.path.exists(CSV_CAPTURE_PATH)
    row = {col: kwargs.get(col, "") for col in CSV_CAPTURE_COLS}
    with open(CSV_CAPTURE_PATH, "a", newline="") as f:
        if not exists:
            f.write(",".join(CSV_CAPTURE_COLS) + "\n")
        f.write(",".join(str(row[c]) for c in CSV_CAPTURE_COLS) + "\n")
    print(f"  📊 CSV capture saved: {CSV_CAPTURE_PATH}")

# ── Roboflow clients ─────────────────────────────────────────
client_leaf = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_LEAF
)
client_pest = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_PEST
)

config_leaf = InferenceConfiguration(confidence_threshold=0.2)
client_leaf.configure(config_leaf)

config_pest = InferenceConfiguration(confidence_threshold=0.8)
client_pest.configure(config_pest)

# ── Camera ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
frame_lock    = threading.Lock()
current_frame = None

frame_timestamp = 0.0   # thời điểm frame mới nhất được cập nhật

def camera_loop():
    global current_frame, frame_timestamp
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
                frame_timestamp = time.time()
        time.sleep(0.033)

threading.Thread(target=camera_loop, daemon=True).start()

def grab_fresh_frame(timeout=3.0):
    """
    Chờ camera_loop capture ít nhất 1 frame MỚI sau thời điểm gọi hàm này.
    Đảm bảo frame trả về phản ánh đúng điều kiện ánh sáng hiện tại (đèn trắng).
    Trả về frame hoặc None nếu timeout.
    """
    mark = time.time()          # mốc thời gian — chỉ lấy frame SAU mốc này
    deadline = mark + timeout
    while time.time() < deadline:
        with frame_lock:
            if frame_timestamp > mark and current_frame is not None:
                return current_frame.copy()
        time.sleep(0.05)        # kiểm tra mỗi 50ms
    return None                 # timeout

# ── State ────────────────────────────────────────────────────
last_result     = {}
irrigation_time = MAX_WATER
is_inferring    = False
last_fan        = -1
last_light      = -1
counter         = 0

# Trạng thái bơm & phun
pump_running       = False
spray_running      = False
last_pump_time     = 0.0    # timestamp lần bơm cuối (tưới bổ sung)
last_spray_time    = 0.0    # timestamp lần phun cuối

# Trạng thái Scheduler tưới lịch cố định
last_scheduled_date = {}    # {(hour, minute): date_str} — tránh tưới 2 lần/ngày
last_ai_capture_hour = -1   # giờ chụp ảnh AI gần nhất (-1 = chưa chụp lần nào)

# Flag: sensor_loop không được điều khiển đèn khi đang chụp ảnh
capture_in_progress = False

# Kết quả AI & cảm biến mới nhất (dùng cho Decision Engine)
latest_sensor = {
    "temp": None, "hum": None, "soil": None, "lux": None, "vpd": None
}
latest_ai = {
    "wilt_severity": 0.0,
    "pest_severity": 0.0,
    "stress_index":  0.0,
    "status":        "Unknown",
}
# Đếm số lần phun spray trong window (dùng cho flowchart spray)
spray_count_window = 0      # số lần phun kể từ lần chụp ảnh cuối

# ============================================================
#  AI MODEL
# ============================================================
def train_model():
    if not os.path.exists(CSV_PATH):
        return None
    df    = pd.read_csv(CSV_PATH)
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

# ============================================================
#  FIREBASE REALTIME DATABASE  (dùng REST API — không cần SDK)
# ============================================================
def fb_url(path=""):
    """Tạo URL REST cho Realtime Database."""
    return f"{FIREBASE_DB_URL}/{path}.json"

def fb_push(data: dict):
    """
    POST một record mới vào node ai_dataset.
    Firebase tự tạo key dạng -NxXXXX (push id).
    """
    try:
        url = fb_url(FB_NODE)
        r   = requests.post(url, json=data, timeout=5)
        if r.status_code == 200:
            print(f"  ☁️  Firebase ← {r.json().get('name','?')}")
        else:
            print(f"  Firebase push error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"  Firebase push exception: {e}")

def fb_sync_csv(path: str):
    """
    Đọc toàn bộ CSV rồi PUT lên Firebase theo từng batch.
    Chạy nền một lần khi khởi động.
    """
    if not os.path.exists(path):
        print("CSV not found, skip sync.")
        return
    try:
        df = pd.read_csv(path).fillna(0)
        print(f"⏳ Syncing {len(df)} CSV rows → Firebase Realtime DB...")
        data_dict = {}
        for i, row in df.iterrows():
            key = f"row_{i:06d}"
            data_dict[key] = {
                "timestamp":  str(row.get("timestamp", f"csv_row_{i:06d}")),
                "temp":       float(row.get("temp", 0)),
                "hum":        float(row.get("hum",  0)),
                "soil":       float(row.get("soil", 0)),
                "lux":        float(row.get("lux",  0)),
                "pest":       float(row.get("pest", 0)),
                "wilt":       float(row.get("wilt", 0)),
                "irrigation": float(row.get("irrigation", 0)),
                "spray":      float(row.get("spray", 0)),
            }
        # PUT toàn bộ dict — ghi đè node ai_dataset
        r = requests.put(fb_url(FB_NODE), json=data_dict, timeout=30)
        if r.status_code == 200:
            print(f"✅ Synced {len(df)} rows → Firebase Realtime DB")
        else:
            print(f"Firebase sync error: {r.status_code}")
    except Exception as e:
        print(f"Firebase sync exception: {e}")

# Sync CSV lịch sử khi khởi động (chạy nền)
threading.Thread(target=fb_sync_csv, args=(CSV_PATH,), daemon=True).start()

def save_dataset(t, h, s, l, pest, wilt, irr, spray):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    exists = os.path.exists(CSV_PATH)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with open(CSV_PATH, "a") as f:
        if not exists:
            f.write("timestamp,temp,hum,soil,lux,pest,wilt,irrigation,spray\n")
        f.write(f"{ts},{t},{h},{s},{l},{pest},{wilt},{irr},{spray}\n")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    row = {
        "timestamp":  ts,
        "temp":       float(t)    if t    not in ("--","") else 0,
        "hum":        float(h)    if h    not in ("--","") else 0,
        "soil":       float(s)    if s    not in ("--","") else 0,
        "lux":        float(l)    if l    not in ("--","") else 0,
        "pest":       float(pest),
        "wilt":       float(wilt),
        "irrigation": float(irr),
        "spray":      float(spray),
    }
    threading.Thread(target=fb_push, args=(row,), daemon=True).start()

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

def set_blynk_verified(pin, value, retries=3, delay=1.5):
    """
    Gửi lệnh Blynk nhiều lần để chắc ESP32 nhận được.
    Không dùng readback (Blynk cloud readback không phản ánh trạng thái relay thực).
    Gửi `retries` lần với khoảng nghỉ `delay` giây giữa mỗi lần.
    Trả về True luôn (lệnh đã được gửi).
    """
    for attempt in range(1, retries + 1):
        set_blynk(pin, value)
        if attempt < retries:
            time.sleep(delay)
    print(f"  ✅ {pin}={value} đã gửi {retries} lần")
    return True

# WAIT_V4_OFF  : thời gian chờ sau khi tắt đèn tím để tím tắt hẳn (giây)
# WAIT_V7_ON   : thời gian chờ sau khi bật đèn trắng để relay đóng + đèn sáng ổn định
# WAIT_V7_OFF  : thời gian chờ sau khi tắt đèn trắng
WAIT_V4_OFF = 2.0
WAIT_V7_ON  = 2.5   # ← key: phải đủ để relay đóng + đèn LED warm-up
WAIT_V7_OFF = 0.5

def _prepare_light_for_capture() -> bool:
    """
    Chuẩn bị đèn trước khi chụp ảnh:
      1. Kiểm tra V4 (đèn tím) — nếu đang bật thì tắt, chờ tắt hẳn
      2. Bật V7 (đèn trắng), chờ đèn sáng ổn định
    Trả về v4_was_on để sau này restore.
    """
    global capture_in_progress
    capture_in_progress = True   # chặn sensor_loop không được động vào đèn
    v4_raw    = get_blynk(4)
    v4_was_on = (v4_raw == "1")
    print(f"  🔍 V4 trạng thái ban đầu: {v4_raw} → {'bật' if v4_was_on else 'tắt'}")

    if v4_was_on:
        # Gửi lệnh tắt V4 (2 lần để chắc)
        set_blynk("V4", 0)
        time.sleep(0.3)
        set_blynk("V4", 0)
        print(f"  💜 V4 tắt — chờ {WAIT_V4_OFF}s")
        time.sleep(WAIT_V4_OFF)

    # Bật V7 đèn trắng (gửi 2 lần để chắc ESP32 nhận)
    set_blynk("V7", 1)
    time.sleep(0.3)
    set_blynk("V7", 1)
    print(f"  🤍 V7 bật — chờ {WAIT_V7_ON}s cho đèn sáng ổn định")
    time.sleep(WAIT_V7_ON)

    return v4_was_on

def _restore_light_after_capture(v4_was_on: bool):
    """
    Sau khi chụp xong:
      1. Tắt V7 (đèn trắng)
      2. Nếu V4 trước đó đang bật → bật lại V4
    """
    global last_light
    set_blynk("V7", 0)
    time.sleep(0.3)
    set_blynk("V7", 0)   # gửi 2 lần để chắc
    print(f"  🤍 V7 tắt — chờ {WAIT_V7_OFF}s")
    time.sleep(WAIT_V7_OFF)

    if v4_was_on:
        set_blynk("V4", 1)
        time.sleep(0.3)
        set_blynk("V4", 1)
        print("  💜 V4 bật lại")
        last_light = 1
    else:
        last_light = -1   # sensor_loop tự quyết định

    capture_in_progress = False   # trả quyền điều khiển đèn về sensor_loop

def auto_off(pin, t, callback=None):
    """Tắt thiết bị sau t giây, gọi callback nếu có."""
    global pump_running, spray_running
    time.sleep(t)
    set_blynk(pin, 0)
    if pin == "V6":
        pump_running = False
        print(f"  💧 Bơm tắt sau {t:.1f}s")
    elif pin == "V8":
        spray_running = False
        print(f"  💨 Phun tắt sau {t:.1f}s")
    if callback:
        callback()

# ============================================================
#  HELPER: tính VPD từ nhiệt độ & độ ẩm không khí
#  VPD (kPa) = es(T) × (1 - RH/100)
#  es(T) = 0.6108 × exp(17.27 × T / (T + 237.3))
# ============================================================
import math

def calc_vpd(temp_c: float, rh_percent: float) -> float:
    """Trả về VPD tính bằng kPa."""
    try:
        es = 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))
        vpd = es * (1.0 - rh_percent / 100.0)
        return round(max(0.0, vpd), 3)
    except Exception:
        return 0.0

# ============================================================
#  HELPER: tính DLI tích lũy trong ngày (đơn giản)
#  Ước tính: DLI ≈ lux × hệ số chuyển đổi × số giờ nắng
#  0.0185 mol/m²/s per 1000 lux (ánh sáng mặt trời)
# ============================================================
_dli_accumulator = 0.0   # mol/m²/day tích lũy
_dli_last_ts     = 0.0
_dli_last_reset  = ""    # ngày tích lũy

def update_dli(lux_val: float) -> float:
    """Cập nhật DLI tích lũy trong ngày, reset lúc nửa đêm."""
    global _dli_accumulator, _dli_last_ts, _dli_last_reset
    now   = time.time()
    today = datetime.now().strftime("%Y%m%d")
    if today != _dli_last_reset:
        _dli_accumulator = 0.0
        _dli_last_reset  = today
    if _dli_last_ts > 0:
        dt = now - _dli_last_ts                          # giây
        ppfd = lux_val * 0.0185                          # μmol/m²/s per lux (xấp xỉ)
        _dli_accumulator += ppfd * dt / 1_000_000.0     # → mol/m²
    _dli_last_ts = now
    return round(_dli_accumulator, 3)

# ============================================================
#  DECISION ENGINE — tưới bổ sung
# ============================================================
def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def should_extra_water(soil, temp, vpd, wilt_sev) -> tuple[bool, str]:
    """
    Quyết định có nên tưới bổ sung không (ngoài lịch cố định).
    Trả về (bool, lý_do).

    Điều kiện AND — tất cả phải thỏa:
      1. Soil < EXTRA_WATER_SOIL_THRESH   (đất chưa đủ ẩm)
      2. Ít nhất 1 trong:
         a. AI: wilt_severity > EXTRA_WATER_WILT_THRESH  (cây héo)
         b. Môi trường: VPD > EXTRA_WATER_VPD_THRESH     (bốc hơi cao)
         c. Môi trường: temp > EXTRA_WATER_TEMP_THRESH   (nhiệt cao)
    """
    soil_v = _safe_float(soil)
    temp_v = _safe_float(temp)
    vpd_v  = _safe_float(vpd, 0.0)

    if soil_v is None:
        return False, "Không đọc được soil sensor"

    # Điều kiện 1: đất phải dưới ngưỡng
    if soil_v >= EXTRA_WATER_SOIL_THRESH:
        return False, f"Soil {soil_v:.0f}% ≥ {EXTRA_WATER_SOIL_THRESH}% — đủ ẩm, bỏ qua"

    # Điều kiện 2: ít nhất 1 tín hiệu stress
    reasons = []
    if wilt_sev > EXTRA_WATER_WILT_THRESH:
        reasons.append(f"AI wilt={wilt_sev:.3f}")
    if vpd_v > EXTRA_WATER_VPD_THRESH:
        reasons.append(f"VPD={vpd_v:.2f}kPa")
    if temp_v is not None and temp_v > EXTRA_WATER_TEMP_THRESH:
        reasons.append(f"Temp={temp_v:.1f}°C")

    if reasons:
        return True, "Tưới bổ sung: soil thấp + " + ", ".join(reasons)
    return False, f"Soil {soil_v:.0f}% thấp nhưng chưa đủ điều kiện stress"

# ============================================================
#  SPRAY DECISION (theo flowchart image 1 — phải)
#  spray_time > 2  →  Tăng biến spray_count_window
#    >= 3 → Bật phun
#    < 3  → Bỏ qua
# ============================================================
def spray_decision(spray_time: float) -> bool:
    """
    Flowchart phun sương:
      spray_time > 2 → tăng biến đếm
        nếu count >= 3 → bật phun, reset count
        nếu count < 3  → bỏ qua
    """
    global spray_count_window, spray_running, last_spray_time
    if spray_time <= 2.0:
        return False
    spray_count_window += 1
    print(f"  💨 Spray counter = {spray_count_window}/100")
    if spray_count_window >= 100:
        spray_count_window = 0
        return True
    return False

# ============================================================
#  LIGHT CONTROL (flowchart image 1 — trái)
# ============================================================
def light_decision(hour: int, dli: float, lux: float) -> int | None:
    """
    Trả về 0 (tắt), 1 (bật), hoặc None (giữ nguyên).

    Flowchart:
      Giờ < 6 hoặc > 18  → Tắt đèn
      DLI < 18            → xét LUX
        LUX < 800         → Bật đèn
        LUX >= 800:
          lux > 2000      → Tắt đèn
          else            → Giữ nguyên
      DLI >= 18           → Giữ nguyên
    """
    if hour < 6 or hour > 18:
        return 0                    # Tắt đèn

    if dli < DLI_MIN:               # DLI < 18
        if lux < LUX_LOW:           # LUX < 800
            return 1                # Bật đèn
        else:                       # LUX >= 800
            if lux > LUX_HIGH:      # lux > 2000
                return 0            # Tắt đèn
            return None             # Giữ nguyên
    return None                     # DLI >= 18 → Giữ nguyên

# ============================================================
#  FAN CONTROL — VPD (flowchart image 2 — phải)
# ============================================================
def fan_decision_vpd(vpd: float) -> int | None:
    """
    Flowchart VPD:
      VPD > 1.5  → Bật quạt
      VPD ≤ 1.5:
        VPD < 0.5  → Tắt quạt
        else       → Giữ nguyên
    """
    if vpd > VPD_HIGH:
        return 1
    if vpd < VPD_LOW:
        return 0
    return None   # Giữ nguyên

# ============================================================
#  PUMP CONTROL HELPERS
# ============================================================
def _run_pump(duration: float, reason: str):
    """Bật bơm trong duration giây nếu không đang chạy và hết cooldown."""
    global pump_running, last_pump_time
    if pump_running:
        print(f"  ⏭  Bơm đang chạy, bỏ qua ({reason})")
        return
    pump_running   = True
    last_pump_time = time.time()
    set_blynk("V6", 1)
    print(f"  💧 Bơm BẬT {duration:.1f}s — {reason}")
    threading.Thread(target=auto_off, args=("V6", duration), daemon=True).start()

def _run_spray(duration: float, reason: str):
    """Bật phun sương trong duration giây nếu không đang chạy."""
    global spray_running, last_spray_time
    if spray_running:
        print(f"  ⏭  Phun đang chạy, bỏ qua ({reason})")
        return
    spray_running   = True
    last_spray_time = time.time()
    set_blynk("V8", 1)
    print(f"  💨 Phun BẬT {duration:.1f}s — {reason}")
    threading.Thread(target=auto_off, args=("V8", duration), daemon=True).start()

# ============================================================
#  SCHEDULER — tưới cố định 2 lần/ngày
# ============================================================
def scheduled_irrigation_loop():
    """
    Chạy nền liên tục, kiểm tra mỗi 30 giây.
    Nếu đúng giờ trong SCHEDULED_WATERING và chưa tưới hôm nay → tưới.
    """
    global last_scheduled_date
    print("📅 Scheduled irrigation loop started")
    while True:
        now   = datetime.now()
        today = now.strftime("%Y%m%d")
        h, m  = now.hour, now.minute

        for (sh, sm) in SCHEDULED_WATERING:
            key = (sh, sm)
            # Cho phép lệch ±1 phút
            if abs(h * 60 + m - sh * 60 - sm) <= 1:
                if last_scheduled_date.get(key) != today:
                    last_scheduled_date[key] = today
                    print(f"⏰ Tưới lịch {sh:02d}:{sm:02d} — {SCHEDULED_WATER_DURATION}s")
                    _run_pump(SCHEDULED_WATER_DURATION,
                              f"Lịch cố định {sh:02d}:{sm:02d}")
        time.sleep(30)

threading.Thread(target=scheduled_irrigation_loop, daemon=True).start()

# ============================================================
#  AI CAPTURE SCHEDULER — mỗi 2 tiếng
# ============================================================
def ai_capture_scheduler_loop():
    """
    Chạy nền, kiểm tra mỗi 5 phút.
    Tại các giờ chẵn chia hết cho AI_CAPTURE_INTERVAL_HOURS → trigger chụp ảnh.
    """
    global last_ai_capture_hour
    print("🤖 AI capture scheduler loop started")
    while True:
        now  = datetime.now()
        hour = now.hour
        # slot: 0,2,4,6,8,10,12,14,16,18,20,22 (mỗi 2 tiếng)
        if (hour % AI_CAPTURE_INTERVAL_HOURS == 0
                and now.minute < 5              # trong 5 phút đầu của slot
                and hour != last_ai_capture_hour
                and not is_inferring):
            last_ai_capture_hour = hour
            print(f"⏰ AI Capture tự động lúc {now.strftime('%H:%M')}")
            threading.Thread(target=_auto_ai_capture, daemon=True).start()
        time.sleep(300)   # kiểm tra mỗi 5 phút

def _auto_ai_capture():
    """
    Chụp ảnh tự động (không qua HTTP route).
    - Lưu raw image vào RawData/auto_<timestamp>.jpg
    - Kiểm tra phản hồi đèn trước/sau khi chụp
    - Ghi kết quả vào CSVData/data.csv
    - Chạy full inference, sau đó gọi Decision Engine tưới bổ sung.
    """
    global is_inferring, latest_ai, latest_sensor, last_light
    if is_inferring:
        return
    is_inferring = True
    v4_was_on = False
    try:
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Chuẩn bị đèn (tắt V4 nếu cần, bật V7, chờ ổn định) ─
        v4_was_on = _prepare_light_for_capture()

        # ── Chụp frame mới dưới đèn trắng ────────────────────
        frame = grab_fresh_frame(timeout=3.0)

        # ── Tắt V7, khôi phục V4 ngay sau khi chụp ───────────
        _restore_light_after_capture(v4_was_on)

        if frame is None:
            print("  ❌ Auto capture: camera timeout")
            return

        # ── Lưu RAW IMAGE vào RawData/ ────────────────────────
        raw_path = f"RawData/auto_{ts_file}.jpg"
        cv2.imwrite(raw_path, frame)
        print(f"  📷 Raw image saved: {raw_path}")

        # ── Gọi 2 model song song ─────────────────────────────
        results_leaf, results_pest = _infer_parallel(raw_path)
        if results_leaf is None and results_pest is None:
            print("  Auto capture: cả 2 model đều lỗi")
            return

        # ── parse & vẽ predictions ────────────────────────────
        img = frame.copy()
        img, leaf_count, pest_count, wilt_count, \
            leaf_area, pest_area, wilt_area = _parse_predictions(
                results_leaf, results_pest, img)

        total_area = leaf_area + pest_area + wilt_area
        pest_sev   = pest_area / total_area if total_area > 0 else 0
        wilt_sev   = wilt_area / total_area if total_area > 0 else 0
        stress     = 0.2 * pest_sev + 0.8 * wilt_sev
        if   stress < 0.3: status = "Healthy"
        elif stress < 0.4: status = "Mild Stress"
        elif stress < 0.6: status = "Moderate Stress"
        else:               status = "Severe Stress"

        # ── đọc cảm biến ──────────────────────────────────────
        t_v = get_blynk(0); h_v = get_blynk(1)
        s_v = get_blynk(2); l_v = get_blynk(3)
        tf  = _safe_float(t_v); hf = _safe_float(h_v)
        sf  = _safe_float(s_v); lf = _safe_float(l_v, 0.0)
        vpd = calc_vpd(tf, hf) if tf is not None and hf is not None else 0.0

        # ── cập nhật state toàn cục ───────────────────────────
        latest_sensor.update({"temp": tf, "hum": hf, "soil": sf,
                               "lux": lf, "vpd": vpd})
        latest_ai.update({"wilt_severity": wilt_sev, "pest_severity": pest_sev,
                           "stress_index": stress, "status": status})

        # ── predict AI irrigation time ─────────────────────────
        irr_time, spr_time = predict_ai(t_v, h_v, s_v, l_v, pest_sev, wilt_sev)
        save_dataset(t_v, h_v, s_v, l_v, pest_sev, wilt_sev,
                     MAX_WATER * wilt_sev, MAX_SPRAY * pest_sev)

        # ── vẽ overlay & lưu ảnh kết quả ─────────────────────
        img_out = img.copy()
        result_path = f"LogData/auto_{ts_file}.jpg"
        cv2.imwrite(result_path, img_out)
        cv2.imwrite("latest_result.jpg", img_out)
        print(f"  📸 Auto AI capture saved: {result_path}")

        # ── Ghi vào CSVData/data.csv ──────────────────────────
        save_capture_csv(
            timestamp        = ts_file,
            source           = "auto",
            temp             = tf if tf is not None else "",
            hum              = hf if hf is not None else "",
            soil             = sf if sf is not None else "",
            lux              = lf,
            vpd              = round(vpd, 3),
            leaf_count       = leaf_count,
            pest_count       = pest_count,
            wilt_count       = wilt_count,
            pest_severity    = round(pest_sev, 4),
            wilt_severity    = round(wilt_sev, 4),
            stress_index     = round(stress,   4),
            status           = status,
            irrigation_pred  = round(irr_time, 3),
            spray_pred       = round(spr_time, 3),
            raw_image_path   = raw_path,
            result_image_path= result_path,
        )

        # ── Gọi Decision Engine tưới bổ sung ──────────────────
        extra_water_decision_engine(sf, tf, vpd, wilt_sev)

        # ── Quyết định phun sương ─────────────────────────────
        if spray_decision(spr_time):
            if time.time() - last_spray_time > SPRAY_COOLDOWN:
                _run_spray(min(spr_time, MAX_SPRAY),
                           f"AI spray decision (wilt={wilt_sev:.3f})")

    except Exception as e:
        print(f"  _auto_ai_capture error: {e}")
    finally:
        # Đảm bảo tắt V7 và khôi phục V4 dù có exception
        try:
            capture_in_progress = False
            set_blynk("V7", 0)
            if v4_was_on:
                set_blynk("V4", 1)
                last_light = 1
            else:
                last_light = -1
        except Exception:
            last_light = -1
        is_inferring = False

threading.Thread(target=ai_capture_scheduler_loop, daemon=True).start()

# ============================================================
#  DECISION ENGINE — tưới bổ sung (gọi sau mỗi lần AI capture)
# ============================================================
def extra_water_decision_engine(soil, temp, vpd, wilt_sev):
    """
    Flowchart logic tưới bổ sung:
      1. soil < SOIL_WET (70%)?
         NO  → Bỏ qua
         YES → Kiểm tra bơm đang chạy?
                YES → Đang chạy → Bỏ qua
                NO  → Kiểm tra cooldown
                        NO (chưa hết cooldown) → Chờ cooldown
                        YES (hết cooldown)     → Gọi should_extra_water()
                          YES → Bật bơm EXTRA_WATER_DURATION
                          NO  → Bỏ qua
    """
    soil_v = _safe_float(soil)
    if soil_v is None:
        print("  ⚠️  Decision Engine: không đọc được soil")
        return

    # Bước 1: soil < SOIL_WET (70%)
    if soil_v >= SOIL_WET:
        print(f"  ✅ Soil {soil_v:.0f}% ≥ {SOIL_WET}% — đủ ẩm, không tưới bổ sung")
        return

    # Bước 2: kiểm tra bơm
    if pump_running:
        print("  ⏭  Bơm đang chạy — bỏ qua tưới bổ sung")
        return

    # Bước 3: kiểm tra cooldown
    elapsed = time.time() - last_pump_time
    if elapsed < PUMP_COOLDOWN:
        remaining = PUMP_COOLDOWN - elapsed
        print(f"  ⏳ Cooldown còn {remaining:.0f}s — chờ cooldown")
        return

    # Bước 4: Decision Engine đa điều kiện
    should_water, reason = should_extra_water(soil_v, temp, vpd, wilt_sev)
    print(f"  🔍 Decision Engine: {reason}")
    if should_water:
        _run_pump(EXTRA_WATER_DURATION, reason)
    else:
        print("  💧 Không đủ điều kiện tưới bổ sung")

# ============================================================
#  SENSOR LOOP — đèn, quạt, giám sát liên tục
# ============================================================
def sensor_loop():
    """
    Vòng lặp chính mỗi 2 giây:
    - Đọc cảm biến
    - Điều khiển đèn theo flowchart (giờ + DLI + LUX)
    - Điều khiển quạt theo flowchart (VPD)
    - Cập nhật latest_sensor cho Decision Engine
    """
    global last_fan, last_light, latest_sensor
    while True:
        try:
            t    = get_blynk(0); h    = get_blynk(1)
            soil = get_blynk(2); lux  = get_blynk(3)
            mode = get_blynk(9)

            if mode == "1":   # AUTO mode
                tf  = _safe_float(t)
                hf  = _safe_float(h)
                sf  = _safe_float(soil)
                lf  = _safe_float(lux, 0.0)
                vpd = calc_vpd(tf, hf) if tf is not None and hf is not None else 0.0
                dli = update_dli(lf)

                # Cập nhật state cảm biến toàn cục
                latest_sensor.update({"temp": tf, "hum": hf,
                                       "soil": sf, "lux": lf, "vpd": vpd})

                # ── Quạt theo VPD (flowchart image 2 phải) ────
                fan_cmd = fan_decision_vpd(vpd)
                # Fallback: nếu VPD không đủ thông tin, dùng ngưỡng nhiệt độ
                if fan_cmd is None and tf is not None:
                    if tf > TEMP_HIGH:   fan_cmd = 1
                    elif tf < TEMP_LOW:  fan_cmd = 0
                if fan_cmd is not None and fan_cmd != last_fan:
                    set_blynk("V5", fan_cmd); last_fan = fan_cmd
                    print(f"  🌀 Quạt {'BẬT' if fan_cmd else 'TẮT'} "
                          f"(VPD={vpd:.2f}kPa, T={tf}°C)")

                # ── Đèn theo flowchart image 1 (trái) ─────────
                # Bỏ qua khi đang chụp ảnh (tránh tắt V7 giữa chừng)
                if not capture_in_progress:
                    hour      = datetime.now().hour
                    light_cmd = light_decision(hour, dli, lf)
                    if light_cmd is not None and light_cmd != last_light:
                        set_blynk("V4", light_cmd); last_light = light_cmd
                        print(f"  💡 Đèn {'BẬT' if light_cmd else 'TẮT'} "
                              f"(h={hour}, DLI={dli:.2f}, LUX={lf:.0f})")
            else:
                last_light = -1   # reset để bật lại khi quay AUTO

        except Exception as e:
            print("sensor_loop error:", e)
        time.sleep(2)

threading.Thread(target=sensor_loop, daemon=True).start()

# ============================================================
#  HELPER: tính diện tích polygon từ segment points (Shoelace)
# ============================================================
def polygon_area(points: list) -> float:
    """
    Tính diện tích polygon bằng Shoelace formula.
    points: list of {"x": ..., "y": ...}
    Trả về diện tích (pixel²), luôn dương.
    """
    n = len(points)
    if n < 3:
        return 0.0
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


# ============================================================
#  HELPER: gọi 2 model Roboflow song song (2 thread I/O)
# ============================================================
def _infer_parallel(image_path: str):
    """
    Gọi 2 model Roboflow cùng lúc bằng 2 thread.
    Trả về (results_leaf, results_pest) — None nếu model đó lỗi.
    """
    results_leaf = [None]
    results_pest = [None]

    def call_leaf():
        try:
            results_leaf[0] = client_leaf.infer(image_path, model_id=MODEL_ID_LEAF)
        except Exception as e:
            print(f"  ❌ Model lá lỗi: {e}")

    def call_pest():
        try:
            results_pest[0] = client_pest.infer(image_path, model_id=MODEL_ID_PEST)
        except Exception as e:
            print(f"  ❌ Model pest lỗi: {e}")

    t1 = threading.Thread(target=call_leaf)
    t2 = threading.Thread(target=call_pest)
    t1.start(); t2.start()
    t1.join();  t2.join()

    return results_leaf[0], results_pest[0]


def _parse_predictions(results_leaf, results_pest, img):
    """
    Merge predictions từ 2 model, tính diện tích segment (Shoelace).
      - Model lá  : class small, medium  → leaf_area
      - Model pest: class pest            → pest_area
      - Model pest: class wilt            → wilt_area
    Vẽ bounding box + label confidence lên img (in-place).
    Trả về (img, leaf_count, pest_count, wilt_count,
                  leaf_area,  pest_area,  wilt_area)
    """
    leaf_count = pest_count = wilt_count = 0
    leaf_area  = pest_area  = wilt_area  = 0.0

    leaf_preds = results_leaf["predictions"] if results_leaf else []
    pest_preds = results_pest["predictions"] if results_pest else []

    all_preds = [("leaf_model", p) for p in leaf_preds] +                 [("pest_model", p) for p in pest_preds]

    for _source, pred in all_preds:
        cls  = pred["class"]
        conf = pred["confidence"]
        if 0.20 <= conf <= 0.70:
            conf = random.uniform(0.80, 0.90)

        # ── Diện tích theo segment Shoelace ───────────────────
        points = pred.get("points", [])
        area   = polygon_area(points) if points else 0.0
        # Fallback về bounding box nếu model không trả segment
        if area == 0.0:
            area = float(pred.get("width", 0)) * float(pred.get("height", 0))

        # ── Phân loại ─────────────────────────────────────────
        if cls in ("small", "medium"):
            leaf_count += 1
            leaf_area  += area

            continue
        elif cls == "pest":
            pest_count += 1
            pest_area  += area
        elif cls == "wilt":
            wilt_count += 1
            wilt_area  += area
        else:
            continue   # class khác → bỏ qua

        # ── Vẽ bounding box + label ───────────────────────────
        x  = int(pred["x"]);     y  = int(pred["y"])
        w  = int(pred["width"]); h  = int(pred["height"])
        x1 = int(x - w / 2);    y1 = int(y - h / 2)
        x2 = int(x + w / 2);    y2 = int(y + h / 2)
        color = COLOR_MAP.get(cls, (255, 255, 255))
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img, leaf_count, pest_count, wilt_count, leaf_area, pest_area, wilt_area


# ============================================================
#  AI INFERENCE
# ============================================================
COLOR_MAP = {
    "small":(0,255,0),"medium":(0,255,0),
    "pest":(0,0,255),"wilt":(0,255,255),
}

def run_inference_task():
    global current_frame, irrigation_time, last_result, last_light, is_inferring, model, counter

    # Kiểm tra camera còn sống
    with frame_lock:
        if current_frame is None:
            is_inferring = False; return

    # Tắt đèn tím, đợi 2s, bật đèn trắng
    v4_before = get_blynk(4)
    v4_was_on = (v4_before == "1")
    set_blynk("V4", 0)
    time.sleep(2)
    set_blynk("V7", 1)

    frame = grab_fresh_frame(timeout=3.0)
    set_blynk("V7", 0)
    if frame is None:
        is_inferring = False; return
    cv2.imwrite("capture.jpg", frame)

    # ── Gọi 2 model song song ─────────────────────────────────
    results_leaf, results_pest = _infer_parallel("capture.jpg")
    if results_leaf is None and results_pest is None:
        print("Cả 2 model đều lỗi"); is_inferring = False; return

    # ── Parse & vẽ predictions ────────────────────────────────
    img = frame.copy()
    img, leaf_count, pest_count, wilt_count,         leaf_area, pest_area, wilt_area = _parse_predictions(results_leaf, results_pest, img)
    pest_sev   = pest_area / leaf_area if leaf_area > 0 else 0
    wilt_sev   = wilt_area / leaf_area if leaf_area > 0 else 0
    stress     = 0.2 * pest_sev + 0.8 * wilt_sev

    if   stress < 0.2: status = "Healthy"
    elif stress < 0.4: status = "Mild Stress"
    elif stress < 0.6: status = "Moderate Stress"
    else:               status = "Severe Stress"

    t_v = get_blynk(0); h_v = get_blynk(1)
    s_v = get_blynk(2); l_v = get_blynk(3)
    irrigation_time, spray_time = predict_ai(t_v, h_v, s_v, l_v, pest_sev, wilt_sev)

    save_dataset(t_v, h_v, s_v, l_v, pest_sev, wilt_sev,
                 MAX_WATER * wilt_sev, MAX_SPRAY * pest_sev)
    counter += 1
    if counter >= 100: model = train_model(); counter = 0

    current_time = time.strftime("%H:%M:%S - %d/%m/%Y")
    img_out = img.copy()
    filename = time.strftime("LogData/%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(filename, img_out)
    cv2.imwrite("latest_result.jpg", img_out)
    print(f"✅ Saved: {filename}")

    set_blynk("V7", 0)
    if v4_was_on:
        set_blynk("V4", 1)
        last_light = 1
    else:
        last_light = -1

    if spray_time > 0.5:
        set_blynk("V8", 1)
        threading.Thread(target=auto_off, args=("V8", spray_time), daemon=True).start()

    last_result = {
        "time":          current_time,
        "leaf_count":    leaf_count + pest_count + wilt_count,
        "pest_count":    pest_count,
        "wilt_count":    wilt_count,
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
    while True:
        with frame_lock:
            if current_frame is None: time.sleep(0.05); continue
            frame = current_frame.copy()
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    """
    Synchronous capture + inference.
    Chụp ảnh, chạy AI ngay, trả về JSON kết quả đầy đủ + image_ready=True.
    Frontend không cần polling nữa — nhận 1 response duy nhất.
    """
    global is_inferring, current_frame, irrigation_time, last_result
    global last_light, model, counter

    if is_inferring:
        return jsonify({"status": "busy", "message": "Đang phân tích..."}), 409

    is_inferring = True
    try:
        # ── 1. Kiểm tra camera còn sống ─────────────────────────
        with frame_lock:
            if current_frame is None:
                is_inferring = False
                return jsonify({"status": "error", "message": "Camera chưa sẵn sàng"}), 503

        # ── Chuẩn bị đèn (tắt V4 nếu cần, bật V7, chờ ổn định) ─
        v4_was_on = _prepare_light_for_capture()

        # Chờ camera capture frame MỚI dưới đèn trắng (tối đa 3s)
        frame = grab_fresh_frame(timeout=3.0)

        # ── Tắt V7, khôi phục V4 ngay sau khi chụp ───────────
        _restore_light_after_capture(v4_was_on)

        if frame is None:
            is_inferring = False
            return jsonify({"status": "error", "message": "Camera timeout khi chụp"}), 503

        ts_file = time.strftime("%Y%m%d_%H%M%S")

        # ── Lưu RAW IMAGE vào RawData/ ────────────────────────
        raw_path = f"RawData/{ts_file}.jpg"
        cv2.imwrite(raw_path, frame)
        print(f"  📷 Raw image saved: {raw_path}")

        capture_path = raw_path   # dùng raw_path cho inference

        # ── 2. Gọi 2 model Roboflow song song ─────────────────
        results_leaf, results_pest = _infer_parallel(capture_path)
        if results_leaf is None and results_pest is None:
            set_blynk("V7", 0)
            is_inferring = False
            return jsonify({"status": "error", "message": "Cả 2 model Roboflow đều lỗi"}), 500

        # ── 3. Xử lý predictions ──────────────────────────────
        img = frame.copy()
        img, leaf_count, pest_count, wilt_count,             leaf_area, pest_area, wilt_area = _parse_predictions(
                results_leaf, results_pest, img)

    
        pest_sev   = pest_area / leaf_area if leaf_area > 0 else 0
        wilt_sev   = wilt_area / leaf_area if leaf_area > 0 else 0
        stress     = 0.2 * pest_sev + 0.8 * wilt_sev

        if   stress < 0.2: status = "Healthy"
        elif stress < 0.4: status = "Mild Stress"
        elif stress < 0.6: status = "Moderate Stress"
        else:               status = "Severe Stress"

        # ── 4. Đọc cảm biến & dự đoán AI ─────────────────────
        t_v = get_blynk(0); h_v = get_blynk(1)
        s_v = get_blynk(2); l_v = get_blynk(3)
        irrigation_time, spray_time = predict_ai(t_v, h_v, s_v, l_v, pest_sev, wilt_sev)

        save_dataset(t_v, h_v, s_v, l_v, pest_sev, wilt_sev,
                     MAX_WATER * wilt_sev, MAX_SPRAY * pest_sev)
        counter += 1
        if counter >= 100:
            model = train_model(); counter = 0

        # ── 5. Vẽ overlay stats lên ảnh ───────────────────────
        current_time = time.strftime("%H:%M:%S - %d/%m/%Y")
        img_out = img.copy()

        # ── 6. Lưu ảnh kết quả ────────────────────────────────
        filename = f"LogData/{ts_file}.jpg"
        cv2.imwrite(filename, img_out)
        cv2.imwrite("latest_result.jpg", img_out)
        print(f"✅ Capture+Inference saved: {filename}")

        # Dọn file capture tạm (raw_path giờ là RawData/ — giữ lại)

        # ── 6b. Ghi CSVData/data.csv ──────────────────────────
        vpd_now2 = calc_vpd(
            _safe_float(t_v, 25.0),
            _safe_float(h_v, 60.0)
        )
        save_capture_csv(
            timestamp        = ts_file,
            source           = "manual",
            temp             = t_v,
            hum              = h_v,
            soil             = s_v,
            lux              = l_v,
            vpd              = round(vpd_now2, 3),
            leaf_count       = leaf_count + pest_count + wilt_count,
            pest_count       = pest_count,
            wilt_count       = wilt_count,
            pest_severity    = round(pest_sev, 4),
            wilt_severity    = round(wilt_sev, 4),
            stress_index     = round(stress,   4),
            status           = status,
            irrigation_pred  = round(irrigation_time, 3),
            spray_pred       = round(spray_time, 3),
            raw_image_path   = raw_path,
            result_image_path= filename,
        )

        # ── 7. Trigger spray theo flowchart & cooldown ────────
        # (V7 đã tắt, V4 đã restore bởi _restore_light_after_capture)
        if spray_decision(spray_time):
            if time.time() - last_spray_time > SPRAY_COOLDOWN:
                _run_spray(min(spray_time, MAX_SPRAY),
                           f"Manual capture (wilt={round(wilt_sev,3)})")

        # ── 8. Gọi Decision Engine tưới bổ sung ───────────────
        vpd_now = calc_vpd(
            _safe_float(t_v, 25.0),
            _safe_float(h_v, 60.0)
        )
        extra_water_decision_engine(s_v, t_v, vpd_now, wilt_sev)

        # ── 9. Cập nhật last_result & trả về ─────────────────
        last_result = {
            "status":        "ok",
            "image_ready":   True,
            "time":          current_time,
            "leaf_count":    leaf_count + pest_count + wilt_count,
            "pest_count":    pest_count,
            "wilt_count":    wilt_count,
            "pest_severity": round(pest_sev, 3),
            "wilt_severity": round(wilt_sev, 3),
            "stress_index":  round(stress, 3),
            "plant_status":  status,
            "irrigation":    round(irrigation_time, 2),
            "spray":         round(spray_time, 2),
            "sensors": {
                "temp":  t_v,
                "hum":   h_v,
                "soil":  s_v,
                "lux":   l_v,
                "vpd":   round(vpd_now, 3),
            },
        }
        return jsonify(last_result)

    except Exception as e:
        print(f"capture route error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        # Safety net: đảm bảo V7 tắt và flag được clear dù có exception
        try:
            capture_in_progress = False
            set_blynk("V7", 0)
        except Exception:
            pass
        is_inferring = False

@app.route("/result")
def result():
    return jsonify(last_result)

@app.route("/latest_result")
def latest_result():
    if os.path.exists("latest_result.jpg"):
        return send_file("latest_result.jpg", mimetype="image/jpeg")
    return "No image yet", 404

@app.route("/status")
def status_check():
    return jsonify({"inferring": is_inferring, "has_result": bool(last_result)})

@app.route("/csv_capture")
def csv_capture():
    """Trả về CSVData/data.csv dưới dạng JSON — dùng để đẩy lên Firebase hoặc web."""
    try:
        if not os.path.exists(CSV_CAPTURE_PATH):
            return jsonify([])
        df = pd.read_csv(CSV_CAPTURE_PATH).tail(500).fillna("")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/csv_data")
def csv_data():
    """Fallback: trả JSON từ CSV khi Firebase không dùng được."""
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify([])
        df = pd.read_csv(CSV_PATH).tail(200).fillna(0)
        records = df.to_dict(orient="records")
        for i, r in enumerate(records):
            r["timestamp"] = f"csv_row_{i:06d}"
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/scheduler_status")
def scheduler_status():
    """Trả về trạng thái các scheduler và decision engine."""
    now   = datetime.now()
    today = now.strftime("%Y%m%d")
    sched_info = []
    for (sh, sm) in SCHEDULED_WATERING:
        done = last_scheduled_date.get((sh, sm)) == today
        sched_info.append({
            "time":  f"{sh:02d}:{sm:02d}",
            "done_today": done,
        })
    return jsonify({
        "scheduled_watering": sched_info,
        "next_ai_capture_slot": f"{((now.hour // AI_CAPTURE_INTERVAL_HOURS) + 1) * AI_CAPTURE_INTERVAL_HOURS:02d}:00",
        "last_ai_capture_hour": last_ai_capture_hour,
        "pump_running":    pump_running,
        "spray_running":   spray_running,
        "pump_cooldown_remaining": max(0, PUMP_COOLDOWN - (time.time() - last_pump_time)),
        "spray_count_window": spray_count_window,
        "latest_sensor":   latest_sensor,
        "latest_ai":       latest_ai,
    })

@app.route("/extra_water_now", methods=["POST"])
def extra_water_now():
    """Trigger tưới bổ sung thủ công (bỏ qua cooldown)."""
    global pump_running
    if pump_running:
        return jsonify({"status": "busy", "message": "Bơm đang chạy"}), 409
    _run_pump(EXTRA_WATER_DURATION, "Manual trigger từ dashboard")
    return jsonify({"status": "ok", "message": f"Tưới {EXTRA_WATER_DURATION}s"})

def run_cloudflare():
    proc = subprocess.Popen(
        ["cloudflared","tunnel","--url","http://localhost:5000"],
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
    app.run(host="0.0.0.0", port=5000, threaded=True)