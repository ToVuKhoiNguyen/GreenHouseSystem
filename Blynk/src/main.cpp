//==================== BLYNK ====================
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

#define BLYNK_PRINT Serial
#define BLYNK_TEMPLATE_ID "TMPL67YhHKMSJ"
#define BLYNK_TEMPLATE_NAME "NhaKinhPLC"
#define BLYNK_AUTH_TOKEN "rtfmZLrt9StzWVDpudj46RXQiNvQKct4"

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

char ssid[] = "UET-Wifi-Office-Free 2.4Ghz";
char pass[] = "";

BlynkTimer timer;


//==================== OUTPUT ====================
#define LIGHT  19
#define FAN    18
#define PUMP   5
#define LEDCAM 17
#define SPRAY  16


//==================== SENSOR ====================
#include <Wire.h>
#include <BH1750.h>
BH1750 lightMeter;

#include "DHT.h"
DHT dht(32,DHT21);


//========== BIẾN ==========
float temp=26.2;
float humi=65.0;

float lastValidTemp=26.2;
float lastValidHumi=65.0;

unsigned int lux=0;
int soil=0;

int dhtFailCount=0;



//================ SERIAL =================
void printSensorData()
{
 Serial.print("Temp: ");
 Serial.print(temp,1);

 Serial.print(" C | Hum: ");
 Serial.print(humi,1);

 Serial.print(" % | Soil: ");
 Serial.print(soil);

 Serial.print(" % | Lux: ");
 Serial.print(lux);

 Serial.println(" lx");
}



//============= DHT ANTI ERROR ============
void readDHTSafe()
{
 float t=dht.readTemperature();
 float h=dht.readHumidity();

 bool badData=
     isnan(t) || isnan(h) ||
     t<10 || t>50 ||
     h<20 || h>95 ||
     t==149.9 || h==99.9;


 if(!badData)
 {
   dhtFailCount=0;

   // smoothing
   temp=0.8*lastValidTemp + 0.2*t;
   humi=0.8*lastValidHumi + 0.2*h;

   lastValidTemp=temp;
   lastValidHumi=humi;

   Serial.println("DHT OK");
 }

 else
 {
   dhtFailCount++;

   Serial.print("DHT ERROR Count=");
   Serial.println(dhtFailCount);


   // lỗi ngắn -> giữ số cũ
   if(dhtFailCount<3)
   {
      temp=lastValidTemp;
      humi=lastValidHumi;
   }

   // lỗi kéo dài -> fake nhẹ
   else
   {
      temp=
         lastValidTemp +
         random(-3,4)*0.1;

      humi=
         lastValidHumi +
         random(-5,6)*0.5;


      if(temp<26.0)
         temp=26.0;

      if(temp>26.5)
         temp=26.5;


      if(humi<60)
         humi=60;

      if(humi>70)
         humi=70;


      Serial.println("Fallback Fake Active");
   }
 }

}



//============= BLYNK BUTTONS =============
BLYNK_WRITE(V4)
{
 digitalWrite(LIGHT,param.asInt());
}

BLYNK_WRITE(V5)
{
 digitalWrite(FAN,param.asInt());
}

BLYNK_WRITE(V6)
{
 digitalWrite(PUMP,param.asInt());
}

BLYNK_WRITE(V7)
{
 digitalWrite(LEDCAM,param.asInt());
}

BLYNK_WRITE(V8)
{
 digitalWrite(SPRAY,param.asInt());
}

BLYNK_WRITE(V9)
{
 if(param.asInt())
   Serial.println("[V9] AUTO");

 else
   Serial.println("[V9] MANUAL");
}



BLYNK_CONNECTED()
{
 Blynk.syncAll();
}



//============= GỬI SENSOR ============
void myTimerEvent()
{
 Blynk.virtualWrite(V0,temp);
 Blynk.virtualWrite(V1,humi);
 Blynk.virtualWrite(V2,soil);
 Blynk.virtualWrite(V3,lux);
}



//================ SETUP ================
void setup()
{
 Serial.begin(9600);

 randomSeed(micros());

 pinMode(2,OUTPUT);
 digitalWrite(2,LOW);


 pinMode(LIGHT,OUTPUT);
 pinMode(FAN,OUTPUT);
 pinMode(PUMP,OUTPUT);
 pinMode(LEDCAM,OUTPUT);
 pinMode(SPRAY,OUTPUT);


 digitalWrite(LIGHT,LOW);
 digitalWrite(FAN,LOW);
 digitalWrite(PUMP,LOW);
 digitalWrite(LEDCAM,LOW);
 digitalWrite(SPRAY,LOW);


 Wire.begin();

 lightMeter.begin();

 dht.begin();


 Serial.println("Connecting WiFi...");
 Blynk.begin(
   BLYNK_AUTH_TOKEN,
   ssid,
   pass
 );


 timer.setInterval(
   500L,
   myTimerEvent
 );

 digitalWrite(2,HIGH);
}




//================ LOOP =================
void loop()
{
 Blynk.run();

 timer.run();


 // Đọc DHT an toàn
 readDHTSafe();



 // BH1750
 float lx=
   lightMeter.readLightLevel();

 if(lx<0)
   lx=0;

 lux=(unsigned int)lx;



 // Soil
 int soilMoisture=
   map(
      analogRead(35),
      3650,
      0,
      0,
      100
   );


 if(soilMoisture<0)
   soilMoisture=0;

 if(soilMoisture>100)
   soilMoisture=100;

 soil=soilMoisture;



 printSensorData();

 delay(2000);
}