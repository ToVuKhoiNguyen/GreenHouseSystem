# Bước 1: Chạy Python server cục bộ trên port 8000
print("Đang chạy Python server trên port 8000...")
server_process = subprocess.Popen(["python", "-m", "http.server", "8000"])

# Đợi 2 giây để server khởi động
time.sleep(2)

# Bước 2: Chạy ngrok trên port 8000
print("Đang chạy ngrok...")
ngrok_process = subprocess.Popen(["ngrok", "http", "8000"])

# Giữ script chạy cho tới khi bạn tắt
try:
    server_process.wait()
    ngrok_process.wait()
except KeyboardInterrupt:
    print("Đang dừng server và ngrok...")
    server_process.terminate()
    ngrok_process.terminate()