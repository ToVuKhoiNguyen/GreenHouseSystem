#define BLYNK_PRINT Serial
#define BLYNK_TEMPLATE_ID "TMPL67YhHKMSJ"
#define BLYNK_TEMPLATE_NAME "NhaKinhPLC"
#define BLYNK_AUTH_TOKEN "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
BlynkTimer timer;

char ssid[] = "UET-Wifi-Office-Free 2.4Ghz";
char pass[] = "";

#define LIGHT  19
#define FAN    18
#define PUMP   5
#define LEDCAM 17
#define SPRAY  16

#include <BH1750.h>
#include <Wire.h>
BH1750 lightMeter;

#include "DHT.h"
#define DHTPIN 32
#define DHTTYPE DHT21
DHT dht(DHTPIN, DHTTYPE);

// Biến toàn cục
float temp = 0, humi = 0;
unsigned int lux = 0;
int soil = 0;

int dhtFailCount = 0;

// ================= BLYNK CONTROL =================
BLYNK_WRITE(V4) { digitalWrite(LIGHT, param.asInt()); }
BLYNK_WRITE(V5) { digitalWrite(FAN, param.asInt()); }
BLYNK_WRITE(V6) { digitalWrite(PUMP, param.asInt()); }
BLYNK_WRITE(V7) { digitalWrite(LEDCAM, param.asInt()); }
BLYNK_WRITE(V8) { digitalWrite(SPRAY, param.asInt()); }

BLYNK_CONNECTED() { 
  Blynk.syncAll();
}

// ================= READ SENSOR =================
void readSensor()
{
  // ---- DHT21 ----
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    Serial.println("DHT ERROR");
    dhtFailCount++;

    // Reset nếu lỗi nhiều lần
    if (dhtFailCount >= 3) {
      Serial.println("Reset DHT...");
      pinMode(DHTPIN, OUTPUT);
      digitalWrite(DHTPIN, HIGH);
      delay(10);
      pinMode(DHTPIN, INPUT);

      dht.begin();
      dhtFailCount = 0;
    }
  } else {
    humi = h;
    temp = t;
    dhtFailCount = 0;
  }

  // ---- Light BH1750 ----
  float lx = lightMeter.readLightLevel();
  if (lx < 0) lx = 0;
  lux = (unsigned int)lx;

  // ---- Soil ----
  int soilMoisture = map(analogRead(35), 3650, 0, 0, 100);
  if (soilMoisture < 0) soilMoisture = 0;
  soil = soilMoisture;

  // Debug
  Serial.print("T: "); Serial.print(temp);
  Serial.print(" | H: "); Serial.print(humi);
  Serial.print(" | Lux: "); Serial.print(lux);
  Serial.print(" | Soil: "); Serial.println(soil);
}

// ================= SEND BLYNK =================
void sendBlynk()
{
  Blynk.virtualWrite(V0, temp);
  Blynk.virtualWrite(V1, humi);
  Blynk.virtualWrite(V2, soil);
  Blynk.virtualWrite(V3, lux);
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  pinMode(2, OUTPUT);
  digitalWrite(2, LOW);

  pinMode(LIGHT, OUTPUT);
  pinMode(FAN, OUTPUT);
  pinMode(PUMP, OUTPUT);
  pinMode(LEDCAM, OUTPUT);
  pinMode(SPRAY, OUTPUT);

  digitalWrite(LIGHT, LOW);
  digitalWrite(FAN, LOW);
  digitalWrite(PUMP, LOW);
  digitalWrite(LEDCAM, LOW);
  digitalWrite(SPRAY, LOW);

  Wire.begin();
  lightMeter.begin();
  dht.begin();

  Serial.println("Connecting to WiFi...");
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  // Timer
  timer.setInterval(2000L, readSensor);   // đọc DHT mỗi 2s
  timer.setInterval(1000L, sendBlynk);    // gửi Blynk
  digitalWrite(2, HIGH);
}

// ================= LOOP =================
void loop() {
  Blynk.run();
  timer.run();
}