//V0 = Temperature
//V1 = Humidity    
//V2 = Soil Moisture
//V3 = Light Intensity
//V4 = Light
//V5 = Fan
//V6 = Pump  
//V7 = LED Camera
//V8 = Spray
//V9 = AUTO/MANUAL
// 35 Soil Sensor
// 32 DHT21 (Temperature and Humidity Sensor)

#define BLYNK_PRINT Serial
#define BLYNK_TEMPLATE_ID "TMPL67YhHKMSJ"
#define BLYNK_TEMPLATE_NAME "NhaKinhPLC"
#define BLYNK_AUTH_TOKEN "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
BlynkTimer timer;
//char ssid[] = "UET-Wifi-Office-Free 2.4Ghz";
//char pass[] = "";
char ssid[] = "Hai";
char pass[] = "........";

#define LIGHT  19
#define FAN    18
#define PUMP   5
#define LEDCAM 17
#define SPRAY  16

#include <BH1750.h>
#include <Wire.h>
BH1750 lightMeter;
#include "DHT.h"
DHT dht(32, DHT21);

float temp=0, humi=0;
unsigned int lux=0;
int soil=0;

BLYNK_WRITE(V4)
{
  int pinValue = param.asInt();
  digitalWrite(LIGHT, pinValue);
}
BLYNK_WRITE(V5)
{
  int pinValue = param.asInt();
  digitalWrite(FAN, pinValue);
}
BLYNK_WRITE(V6)
{
  int pinValue = param.asInt();
  digitalWrite(PUMP, pinValue);
}
BLYNK_WRITE(V7)
{
  int pinValue = param.asInt();
  digitalWrite(LEDCAM, pinValue);
}
BLYNK_WRITE(V8)
{
  int pinValue = param.asInt();
  digitalWrite(SPRAY, pinValue);
}
BLYNK_CONNECTED() { 
  //Blynk.syncVirtual(V1); 
  Blynk.syncAll();
}
void myTimerEvent()
{
  Blynk.virtualWrite(V0, temp);
  Blynk.virtualWrite(V1, humi);
  Blynk.virtualWrite(V2, soil);
  Blynk.virtualWrite(V3, lux);
}
void setup() {
  Serial.begin(9600);
  pinMode(2, OUTPUT); digitalWrite(2, 0);
  pinMode(LIGHT, OUTPUT); digitalWrite(LIGHT, 0);
  pinMode(FAN, OUTPUT); digitalWrite(FAN, 0);
  pinMode(PUMP, OUTPUT); digitalWrite(PUMP, 0);
  pinMode(LEDCAM, OUTPUT); digitalWrite(LEDCAM, 0);
  pinMode(SPRAY, OUTPUT); digitalWrite(SPRAY, 0);

  Serial.println("Connecting to WiFi...");
  Wire.begin();
  lightMeter.begin();
  dht.begin();

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  timer.setInterval(500L, myTimerEvent);
  digitalWrite(2, 1);

}

void loop() {
  Blynk.run();
  timer.run();

  humi = dht.readHumidity();
  temp = dht.readTemperature();
  float lx = lightMeter.readLightLevel();
  if (lx < 0) lx=0;
  lux = (unsigned int)lx;

  int soilMoisture = map(analogRead(35),3650,0,0,100);
  if (soilMoisture < 0) soilMoisture = 0;
  soil = soilMoisture;

  Serial.print(temp);
  Serial.print("\t");
  Serial.print(humi);
  Serial.print("\t");
  Serial.print(lux);
  Serial.print("\t");
  Serial.print(soil);
  Serial.println(" ");
  delay(2e3);

}
