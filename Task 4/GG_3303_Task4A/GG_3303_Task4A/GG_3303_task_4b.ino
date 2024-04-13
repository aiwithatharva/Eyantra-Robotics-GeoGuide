/*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This file is to implement Task 4B of Geo Guide (GG) Theme (eYRC 2023-24) using ESP32 on Arduino IDE.
*****************************************************************************************
*/

// Team ID:			[ GG_3303 ]
// Author List:		[ Atharva Satish Attarde, Prachit Suresh Deshinge, Ashutosh Anil Dongre, Nachiket Ganesh Apte ]
// Filename:			task_4b.ino



#include <WiFi.h>
#include <WiFiUdp.h>

const char *ssid = "vivo T1 5G";
const char *password = "11111111";

const int port = 8002;

char latestStop;
char latestDirection;
WiFiUDP udp;
int count = 0;

// Define pin numbers for various components
#define LEDRed 13
#define LEDGreen 12
#define Buzzer 23
#define MotorClockLeft 16
#define MotorClockRight 4
#define MotorAntiClockLeft 2
#define MotorAntiClockRight 17
#define MotorLeftSpeed 5
#define MotorRightSpeed 15
#define IRSensorFarLeft 19
#define IRSensorLeft 21
#define IRSensorFarRight 18
#define IRSensorRight 22


/**
 * Sets up the initial state of the device. This includes initializing serial communication,
 * setting pin modes, connecting to WiFi, and starting UDP listening.
 */
void setup() {
    Serial.begin(115200);
    pinMode(LEDGreen, OUTPUT);
    pinMode(LEDRed, OUTPUT);
    pinMode(Buzzer, OUTPUT);
    pinMode(MotorClockLeft, OUTPUT);
    pinMode(MotorClockRight, OUTPUT);
    pinMode(MotorAntiClockLeft, OUTPUT);
    pinMode(MotorAntiClockRight, OUTPUT);
    pinMode(MotorLeftSpeed, OUTPUT);
    pinMode(MotorRightSpeed, OUTPUT);
    pinMode(IRSensorFarLeft, INPUT);
    pinMode(IRSensorFarRight, INPUT);
    pinMode(IRSensorRight, INPUT);
    pinMode(IRSensorLeft, INPUT);
    Serial.println();
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);

    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());

    // Start listening for UDP packets
    udp.begin(port);
    Serial.print("Listening on UDP port ");
    Serial.println(port);
}


/**
 * Moves the robot forward by setting appropriate motor speeds and directions.
 */
void forward()
{
    digitalWrite(MotorClockLeft, HIGH);
    digitalWrite(MotorClockRight, HIGH);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, HIGH);
    digitalWrite(MotorRightSpeed, HIGH);
}

/**
 * Turns the robot to the left by adjusting motor speeds and directions.
 */
void left()
{
    digitalWrite(MotorClockLeft, LOW);
    digitalWrite(MotorClockRight, HIGH);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, LOW);
    digitalWrite(MotorRightSpeed, HIGH);  
}


/**
 * Turns the robot to the right by adjusting motor speeds and directions.
 */
void right()
{
    digitalWrite(MotorClockLeft, HIGH);
    digitalWrite(MotorClockRight, LOW);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, HIGH);
    digitalWrite(MotorRightSpeed, LOW);
}

/**
 * Emits a beep from the buzzer for 1 second.
 */
void buzzerBeep_1s()
{
    digitalWrite(Buzzer, LOW);
    delay(1000);
    digitalWrite(Buzzer, HIGH);
}

/**
 * Emits a beep from the buzzer for 5 seconds and turns on the LED.
 */
void buzzerBeep_5s_and_Led()
{
    digitalWrite(Buzzer, LOW);
    digitalWrite(LEDRed, HIGH);
    delay(5000);
    digitalWrite(Buzzer, HIGH);
    digitalWrite(LEDRed, LOW);
}

/**
 * Emits a beep from the buzzer for 1 second and turns on the LED.
 */
void buzzerBeep_1s_and_Led()
{
    digitalWrite(LEDRed, HIGH);
    digitalWrite(Buzzer, LOW);
    delay(1000);
    digitalWrite(LEDRed, LOW);
    digitalWrite(Buzzer, HIGH);
}


/**
 * Stops all motor activity, effectively halting the robot.
 */
void stop()
{
    digitalWrite(MotorClockLeft, LOW);
    digitalWrite(MotorClockRight, LOW);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, LOW);
    digitalWrite(MotorRightSpeed, LOW);
}


/**
 * Determines the robot's next action based on the last stop and direction received.
 */
void Stop_And_Turn()
{

    digitalWrite(MotorClockLeft, LOW);
    digitalWrite(MotorClockRight, LOW);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, LOW);
    digitalWrite(MotorRightSpeed, LOW);

    if (latestStop == 'A' && latestDirection == 'N')
    {
        buzzerBeep_1s();
        left();
        delay(100);
        forward();
        delay(300);
        return;
    }
    else if(latestStop == 'B' && latestDirection == 'N')
    {
        buzzerBeep_1s();
        forward();
        delay(300);
        return;
    }
    else if(latestStop == 'C' && latestDirection == 'N')
    {
        buzzerBeep_1s();
        turnRight();
        return;
    }
    else if(latestStop == 'D' && latestDirection == 'E')
    {
        buzzerBeep_1s();
        turnLeft();
        return;
    }
    else if(latestStop == 'E' && latestDirection == 'N')
    {
        buzzerBeep_1s();
        turnRight();
        return;
    }
    else if(latestStop == 'F' && latestDirection == 'E')
    {
        buzzerBeep_1s();
        turnRight();
        right();
        delay(100);
        return;
    }
    else if(latestStop == 'G' && latestDirection == 'S')
    {
        buzzerBeep_1s();
        forward();
        delay(300);
        return;
    }
    else if(latestStop == 'H' && latestDirection == 'S')
    {
        buzzerBeep_1s();
        turnRight();
        return;
    }
    else if(latestStop == 'I' && latestDirection == 'W')
    {
        buzzerBeep_1s();
        forward();
        delay(300);
        return;
    }
    else if(latestStop == 'B' && latestDirection == 'W')
    {
        turnLeft();
        return;
    }
    else if(latestStop == 'A' && latestDirection == 'S')
    {
        buzzerBeep_1s();
        forward();
        delay(300);
        right();
        delay(250);
        return;
    }
}


/**
 * Handles incoming UDP packets and updates the latest stop and direction variables.
 */
void recievingInputs()
{
    int packetSize = udp.parsePacket();
    if (packetSize) {
        char buffer[255];
        int len = udp.read(buffer, 255);
        if (len > 0) {
        buffer[len] = '\0';
        }
        Serial.print("Received: ");
        Serial.print(buffer[0]);
        Serial.println(buffer[1]);
        Serial.println(buffer[1]);
        latestStop = buffer[0];
        latestDirection = buffer[1];
    }
}


/**
 * Performs a left turn maneuver until the right IR sensor detects an obstacle.
 */
void turnLeft()
{
    left();
    delay(800);
    while(digitalRead(IRSensorRight) == LOW)
    {
        left();
    }
}


/**
 * Performs a right turn maneuver until the left IR sensor detects an obstacle.
 */
void turnRight()
{
    right();
    delay(800);
    while(digitalRead(IRSensorLeft) == LOW)
    {
        right();
    }
}


/**
 * Controls the robot's movement based on sensor data, adjusting direction as needed.
 */
void NormalMovement(int leftSensorData, int rightSensorData, int farleftSensorData, int farrightSensorData)
{
    if (leftSensorData == HIGH && rightSensorData == LOW || farleftSensorData == LOW && farrightSensorData == HIGH)
    {
        left();
    }
    else if(leftSensorData == LOW && rightSensorData == HIGH || farleftSensorData == HIGH && farrightSensorData == LOW)
    {   

        right();
    }
    else
    {
        forward();
    }
}


/**
 * Executes the ending sequence, which includes stopping the robot and emitting a long beep with LED indication.
 */
void ending()
{
    stop();
    buzzerBeep_5s_and_Led();
    delay(10000);
}


/**
 * Main loop handling robot movement and decision-making based on sensor
input and received commands.
*/
void loop() 
{   
    if (count == 0)
    {   stop();
        digitalWrite(Buzzer, HIGH);
        delay(60000); // Wait for a minute till the person turning on the robot switch vacate the area.
        buzzerBeep_1s_and_Led();
        count = 1;
    }
    recievingInputs();
    int leftSensorData = digitalRead(IRSensorLeft);
    int rightSensorData = digitalRead(IRSensorRight);
    int farleftSensorData = digitalRead(IRSensorFarLeft);
    int farrightSensorData = digitalRead(IRSensorFarRight);
    if(leftSensorData == HIGH && rightSensorData == HIGH)
    {
        Stop_And_Turn();
    }
    else
    {
        NormalMovement(leftSensorData, rightSensorData, farleftSensorData, farrightSensorData);
    }

    if(latestStop == 'X' && latestDirection == 'S')
    {
        ending();
    }

}
