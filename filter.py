import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("ai_dataset.csv")

# FIX timestamp format (QUAN TRỌNG)
df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='%Y%m%d_%H%M%S',
    errors='coerce'
)

# drop lỗi
df = df.dropna(subset=['timestamp'])

# sort + set index
df = df.sort_values('timestamp')
df = df.set_index('timestamp')

cols = ['temp', 'hum', 'soil', 'lux']

# =========================
# 2. RANGE FILTER
# =========================
def range_filter(df):
    df = df.copy()
    df.loc[(df['temp'] < 5) | (df['temp'] > 35), 'temp'] = np.nan
    df.loc[(df['hum'] < 30) | (df['hum'] > 100), 'hum'] = np.nan
    df.loc[(df['soil'] < 20) | (df['soil'] > 100), 'soil'] = np.nan
    df.loc[(df['lux'] < 0) | (df['lux'] > 5000), 'lux'] = np.nan
    return df

df = range_filter(df)

# =========================
# 3. INTERPOLATE (lấp NaN)
# =========================
df[cols] = df[cols].interpolate(method='time')

# =========================
# 4. FILTERS (TỐI ƯU)
# =========================

# Moving Average
def moving_average(series, window=5):
    return series.rolling(window, min_periods=1).mean()

# Median Filter
def median_filter(series, window=3):
    return series.rolling(window, center=True, min_periods=1).median()

# Hampel Filter (vectorized nhanh hơn)
def hampel_filter(series, window=5, k=3):
    rolling_median = series.rolling(window, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = diff.rolling(window, center=True).median()

    threshold = k * 1.4826 * mad
    outlier = diff > threshold

    filtered = series.copy()
    filtered[outlier] = rolling_median[outlier]

    return filtered

# EMA
def ema(series, alpha=0.3):
    return series.ewm(alpha=alpha, adjust=False).mean()

# =========================
# 5. APPLY FILTERS
# =========================
results = {}

for col in cols:
    raw = df[col]

    results[col] = {
        "Raw": raw,
        "MA": moving_average(raw),
        "Median": median_filter(raw),
        "Hampel": hampel_filter(raw),
        "EMA": ema(raw)
    }

# =========================
# 6. METRICS
# =========================
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def mae(a, b):
    return np.mean(np.abs(a - b))

def smoothness(x):
    return np.sum(np.abs(np.diff(x)))

def correlation(a, b):
    return np.corrcoef(a, b)[0,1]

# reference signal (giả lập ground truth)
def reference_signal(x):
    return x.rolling(10, min_periods=1).mean()

# =========================
# 7. EVALUATION
# =========================
evaluation = []

for col in cols:
    ref = reference_signal(results[col]["Raw"])

    for method in ["MA", "Median", "Hampel", "EMA"]:
        filt = results[col][method]

        mask = (~ref.isna()) & (~filt.isna())

        r = rmse(ref[mask], filt[mask])
        m = mae(ref[mask], filt[mask])
        s = smoothness(filt[mask])
        c = correlation(ref[mask], filt[mask])

        evaluation.append([col, method, r, m, s, c])

eval_df = pd.DataFrame(evaluation, columns=[
    "Variable", "Filter", "RMSE", "MAE", "Smoothness", "Correlation"
])

print("\n=== EVALUATION RESULT ===")
print(eval_df)

# =========================
# 8. PLOT
# =========================
for col in cols:
    plt.figure(figsize=(10,5))
    plt.plot(results[col]["Raw"], label="Raw", alpha=0.4)

    for method in ["MA", "Median", "Hampel", "EMA"]:
        plt.plot(results[col][method], label=method)

    plt.title(f"{col} filtering comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# 9. SAVE
# =========================
eval_df.to_csv("evaluation.csv", index=False)

# lưu dữ liệu đã lọc (EMA là output chính)
filtered_df = df.copy()
for col in cols:
    filtered_df[col] = results[col]["EMA"]

filtered_df.to_csv("filtered_data.csv")

print("\nDONE!")