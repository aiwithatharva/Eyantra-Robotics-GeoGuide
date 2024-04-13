/*
 * Team Id: GG_3303
 * Author List: Atharva Satish Attarde, Prachit Suresh Deshinge, Ashutosh Anil Dongre, Nachiket Ganesh Apte
 * Filename: Task_6.ino
 * Theme: Geo Guide
 * Functions: setup(), setupWiFi(), setupPins(), turnright(), turnleft(), receivingInputs(), forward(), left1(), left2(), right1(), right2(),
 * buzzerBeep_1s(), buzzerBeep_5s(), buzzerBeep_1s_and_Led(), stop(), rotate(), stopRoutine(), ending(), NormalMovement(), movement(), loop()
 *
 * Global Variables: ssid, password, localUdpPort,latestStop, latestDirection, nextStop, maskTemp, stoparr[], LEDRed, LEDGreen, Buzzer, MotorClockLeft,
 *                   MotorClockRight, MotorAntiClockLeft, MotorAntiClockRight, MotorLeftSpeed, MotorRightSpeed, IRSensorFarLeft, IRSensorLeft,
 *                   IRSensorFarRight, IRSensorRight, IRSensorMid, rotate_cout_with_ids, packetreceived, sensorLeftHigh, sensorMidHigh ,
 *                   sensorRightHigh, startTime, durationThreshold
 *
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <unordered_map>
const char *ssid = "abc";
const char *password = "11111111";

WiFiUDP udp;

unsigned int localUdpPort = 4210; 
char latestStop;
char latestDirection;
char nextStop;
int maskTemp;
int stoparr[5] = {0, 0, 0, 0, 0};
const int Buzzer = 13;
const int MotorClockLeft = 16;
const int MotorClockRight = 4;
const int MotorAntiClockLeft = 2;
const int MotorAntiClockRight = 17;
const int MotorLeftSpeed = 5;   
const int MotorRightSpeed = 15; 
const int IRSensorFarLeft = 19;
const int IRSensorLeft = 21;
const int IRSensorFarRight = 18;
const int IRSensorRight = 22;
const int IRSensorMid = 23;
std::unordered_map<char, int> rotate_cout_with_ids; 
int packetreceived = 0;
bool sensorLeftHigh = false;
bool sensorMidHigh = false;
bool sensorRightHigh = false;
unsigned long startTime = 0;
unsigned long durationThreshold = 0;

/**
 * Function Name: setup
 * Input: None
 * Output: None
 * Logic: Initializes the system setup including buzzer activation, serial communication, pin configuration, and WiFi setup.
 *        This function sets up the system by activating the buzzer, initializing serial communication,
 *        configuring pins using setupPins(), and establishing a WiFi connection using setupWiFi().
 *        It also starts listening for UDP packets on the specified port and prints the port information to the Serial monitor.
 * Example Call: setup();
 */

void setup()
{
    digitalWrite(Buzzer, HIGH);
    Serial.begin(115200);
    setupPins();
    setupWiFi();
    udp.begin(localUdpPort);
    Serial.print("Listening on UDP port ");
    Serial.println(localUdpPort);
}

/**
 * Function Name: setupWiFi
 * Input: None
 * Output: None
 * Logic: Connects to the WiFi network using the provided SSID and password.
 *        Prints connection progress to the Serial monitor and displays the assigned IP address upon successful connection.
 * Example Call: setupWiFi();
 */

void setupWiFi()
{
    Serial.println();
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);

    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
}

/**
 * Function Name: setupPins
 * Input: None
 * Output: None
 * Logic: Configures the pins for various components, setting them as OUTPUT or INPUT as necessary.
 *        Uncomment or add pinMode lines based on the specific components used in your project.
 * Example Call: setupPins();
 */

void setupPins()
{
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
    pinMode(IRSensorMid, INPUT);
}

/**
 * Function Name: turnright
 * Input: None
 * Output: None
 * Logic: Initiates a right turn by calling the right1() function and continues the turn until the left IR sensor is activated.
 *        A delay of 500 milliseconds is inserted to allow the turn to initiate before checking the sensor.
 *        If the latest stop is 'j' and the next stop is 'k', the function returns without executing the turn.
 * Example Call: turnright();
 */

void turnright()
{
    if (latestStop == 'j' && nextStop == 'k')
        return;
    right1();
    delay(500);
    while (digitalRead(IRSensorLeft) == LOW)
    {
        right1();
    }
}

/**
 * Function Name: turnleft
 * Input: None
 * Output: None
 * Logic: Initiates a left turn by calling the left1() function and continues the turn until the right IR sensor is activated.
 *        A delay of 500 milliseconds is inserted to allow the turn to initiate before checking the sensor.
 * Example Call: turnleft();
 */

void turnleft()
{

    left1();
    delay(500);
    while (digitalRead(IRSensorRight) == LOW)
    {
        left1();
    }
}

/**
 * Function Name: receivingInputs
 * Input: None
 * Output: None
 * Logic: Listens for incoming UDP packets and processes the received data.
 *        Parses the packet to extract information such as the latest stop, next stop, direction, and a masking value.
 *        Updates global variables (latestStop, nextStop, latestDirection, and stoparr) accordingly.
 * Example Call: receivingInputs();
 */

void receivingInputs()
{
    int packetSize = udp.parsePacket();
    if (packetSize)
    {
        packetreceived = 1;
        char buffer[512];
        int len = udp.read(buffer, 512);
        if (len > 0)
        {
            buffer[len] = '\0';
            char char1 = buffer[0];
            char char2 = buffer[2];
            char character = buffer[4];
            maskTemp = int(buffer[5]) - 48;
            Serial.print("First char: ");
            Serial.println(char1);
            Serial.print("Second char: ");
            Serial.println(char2);
            Serial.print("Character: ");
            Serial.println(character);

            latestStop = char1;
            nextStop = char2;
            latestDirection = character;
            if (maskTemp == 1)
            {
                stoparr[(int)nextStop - 65] = 1;
            }
        }
    }
}

/**
 * Function Name: forward
 * Input: None
 * Output: None
 * Logic: Initiates forward motion by activating the clockwise rotation of both left and right motors.
 *        Adjusts motor speed by setting the corresponding speed pins to HIGH.
 * Example Call: forward();
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
 * Function Name: left1
 * Input: None
 * Output: None
 * Logic: Initiates a left turn by stopping the left motor and activating the right motor in a clockwise direction.
 *        Adjusts motor speed to control the turn by setting the corresponding speed pins accordingly.
 * Example Call: left1();
 */

void left1()
{
    digitalWrite(MotorClockLeft, LOW);
    digitalWrite(MotorClockRight, HIGH);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, LOW);
    digitalWrite(MotorRightSpeed, HIGH);
}

/**
 * Function Name: left2
 * Input: None
 * Output: None
 * Logic: Initiates a sharper left turn by stopping the left motor and activating both the right motor in a clockwise direction
 *        and the left motor in an anti-clockwise direction.
 *        Adjusts motor speed to control the turn by setting the corresponding speed pins accordingly.
 * Example Call: left2();
 */

void left2()
{
    digitalWrite(MotorClockLeft, LOW);
    digitalWrite(MotorClockRight, HIGH);
    digitalWrite(MotorAntiClockLeft, HIGH);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, HIGH);
    digitalWrite(MotorRightSpeed, HIGH);
}

/**
 * Function Name: right1
 * Input: None
 * Output: None
 * Logic: Initiates a right turn by activating the left motor in a clockwise direction and stopping the right motor.
 *        Adjusts motor speed to control the turn by setting the corresponding speed pins accordingly.
 * Example Call: right1();
 */

void right1()
{
    digitalWrite(MotorClockLeft, HIGH);
    digitalWrite(MotorClockRight, LOW);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, LOW);
    digitalWrite(MotorLeftSpeed, HIGH);
    digitalWrite(MotorRightSpeed, LOW);
}

/**
 * Function Name: right2
 * Input: None
 * Output: None
 * Logic: Initiates a sharper right turn by activating both the left motor in a clockwise direction and the right motor in an anti-clockwise direction.
 *        Adjusts motor speed to control the turn by setting the corresponding speed pins accordingly.
 * Example Call: right2();
 */

void right2()
{
    digitalWrite(MotorClockLeft, HIGH);
    digitalWrite(MotorClockRight, LOW);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, HIGH);
    digitalWrite(MotorLeftSpeed, HIGH);
    digitalWrite(MotorRightSpeed, HIGH);
}

/**
 * Function Name: buzzerBeep_1s
 * Input: None
 * Output: None
 * Logic: Activates the buzzer to produce a beep for 1 second by setting the Buzzer pin LOW for the specified duration and then HIGH.
 * Example Call: buzzerBeep_1s();
 */

void buzzerBeep_1s()
{
    digitalWrite(Buzzer, LOW);
    delay(1000);
    digitalWrite(Buzzer, HIGH);
}

/**
 * Function Name: buzzerBeep_5s
 * Input: None
 * Output: None
 * Logic: Activates the buzzer to produce a beep for 5 seconds by setting the Buzzer pin LOW for the specified duration and then HIGH.
 * Example Call: buzzerBeep_5s();
 */

void buzzerBeep_5s()
{
    digitalWrite(Buzzer, LOW);
    delay(5000);
    digitalWrite(Buzzer, HIGH);
}

/**
 * Function Name: buzzerBeep_1s_and_Led
 * Input: None
 * Output: None
 * Logic: Activates the buzzer to produce a 1-second beep and may control an LED (commented out in the provided code).
 *        Set the LED pin configurations as needed.
 * Example Call: buzzerBeep_1s_and_Led();
 */

void buzzerBeep_1s_and_Led()
{
    digitalWrite(Buzzer, LOW);
    delay(1000);
    digitalWrite(Buzzer, HIGH);
}

/**
 * Function Name: stop
 * Input: None
 * Output: None
 * Logic: Stops all motors and sets the motor speed pins to LOW, halting the motion.
 * Example Call: stop();
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
 * Function Name: rotate
 * Input: None
 * Output: None
 * Logic: Initiates a rotation by activating the left motor in a clockwise direction and the right motor in an anti-clockwise direction.
 *        Adjusts motor speed to control the rotation and introduces a delay for the rotation duration.
 *        If the latest stop is 'C', a brief additional delay is introduced.
 *        Continues the rotation until the far-right IR sensor is activated and then stops.
 * Example Call: rotate();
 */

void rotate()
{
    Serial.println("Rotating...");
    digitalWrite(MotorClockLeft, HIGH);
    digitalWrite(MotorClockRight, LOW);
    digitalWrite(MotorAntiClockLeft, LOW);
    digitalWrite(MotorAntiClockRight, HIGH);
    digitalWrite(MotorLeftSpeed, HIGH);
    digitalWrite(MotorRightSpeed, HIGH);
    delay(900);
    if (latestStop == 'C')
    {
        delay(60);
    }
    while (digitalRead(IRSensorFarRight) != HIGH)
    {
        right1();
    }
    stop();
}

/**
 * Function Name: stopRoutine
 * Input: None
 * Output: None
 * Logic: Initiates a stop routine by calling the stop() function to halt all motors,
 *        followed by a 1-second beep using the buzzerBeep_1s() function.
 * Example Call: stopRoutine();
 */

void stopRoutine()
{
    stop();
    buzzerBeep_1s();
}

/**
 * Function Name: ending
 * Input: None
 * Output: None
 * Logic: Initiates an ending routine by calling the stop() function to halt all motors,
 *        followed by a 5-second beep using the buzzerBeep_5s() function.
 * Example Call: ending();
 */

void ending()
{
    stop();
    buzzerBeep_5s();
}

/**
 * Function Name: j_k_NormalMovement
 * Input: None
 * Output: None
 * Logic: Implements the normal movement behavior for stops 'J' and 'K' based on infrared sensor data.
 *        If the left sensor is HIGH and the middle sensor is LOW, initiates a left turn using left1().
 *        If the left sensor is LOW and the middle sensor is HIGH, initiates a right turn using right1().
 *        If neither condition is met, moves forward using the forward() function.
 * Example Call: j_k_NormalMovement();
 */

void j_k_NormalMovement()
{
    int leftSensorData = digitalRead(IRSensorLeft);
    int midSensorData = digitalRead(IRSensorMid);
    if (leftSensorData == HIGH && midSensorData == LOW)
    {
        left1();
    }
    else if (leftSensorData == LOW && midSensorData == HIGH)
    {
        {
            right1();
        }
    }
    else
    {
        forward();
    }
}

/**
 * Function Name: NormalMovement
 * Input: None
 * Output: None
 * Logic: Implements the normal movement behavior based on the latest and next stops and infrared sensor data.
 *        If the current movement involves stops 'J' and 'K', calls j_k_NormalMovement().
 *        Analyzes infrared sensor data to determine the appropriate movement:
 *          - If the left sensor is HIGH and the right sensor is LOW, or the far-left sensor is LOW and the far-right sensor is HIGH,
 *            initiates a left turn. For stop 'E' to 'd', it uses left2() otherwise left1().
 *          - If the left sensor is LOW and the right sensor is HIGH, or the far-left sensor is HIGH and the far-right sensor is LOW,
 *            initiates a right turn. For stop 'd' to 'E', it uses right2() otherwise right1().
 *          - If none of the above conditions are met, moves forward.
 * Example Call: NormalMovement();
 */

void NormalMovement()
{
    if (latestStop == 'j' && nextStop == 'k')
    {
        j_k_NormalMovement();
    }

    int leftSensorData = digitalRead(IRSensorLeft);
    int rightSensorData = digitalRead(IRSensorRight);
    int midSensorDate = digitalRead(IRSensorMid);
    int farleftSensorData = digitalRead(IRSensorFarLeft);
    int farrightSensorData = digitalRead(IRSensorFarRight);

    if (latestStop == 'k' && nextStop == 'd')
    {
        farleftSensorData = LOW;
        farrightSensorData = LOW;
    }
    if (leftSensorData == HIGH && rightSensorData == LOW || farleftSensorData == LOW && farrightSensorData == HIGH)
    {
        if (latestStop == 'E' && nextStop == 'd')
        {
            left2();
        }
        else
        {
            left1();
        }
    }
    else if (leftSensorData == LOW && rightSensorData == HIGH || farleftSensorData == HIGH && farrightSensorData == LOW)
    {
        if (latestStop == 'd' && nextStop == 'E')
        {
            right2();
        }
        else
        {
            right1();
        }
    }
    else
    {
        forward();
    }
}

/**
 * Function Name: movement
 * Input: None
 * Output: None
 * Logic: Implements the overall movement logic based on infrared sensor data and stop conditions.
 *        Records the start time when both left and right sensors are HIGH.
 *        Initiates specific movements based on the duration threshold and the latest direction.
 *        Handles special cases for stops 'A' to 'D', 'E', and 'x'.
 * Example Call: movement();
 */

void movement()
{
    sensorLeftHigh = digitalRead(IRSensorLeft) == HIGH;
    sensorRightHigh = digitalRead(IRSensorRight) == HIGH;
    sensorMidHigh = digitalRead(IRSensorMid) == HIGH;
    if (sensorLeftHigh && sensorRightHigh && startTime == 0)
    {
        startTime = millis(); 
    }
    if (sensorLeftHigh && sensorRightHigh && (millis() - startTime >= durationThreshold))
    {
        switch (latestDirection)
        {
        case 'F':
            if ((digitalRead(IRSensorLeft) && digitalRead(IRSensorRight) && digitalRead(IRSensorMid)) == HIGH)
                forward();
            break;
        case 'R':
            if ((digitalRead(IRSensorLeft) && digitalRead(IRSensorRight) && digitalRead(IRSensorMid)) == HIGH)
                turnright();
            break;
        case 'L':
            if ((digitalRead(IRSensorLeft) && digitalRead(IRSensorRight) && digitalRead(IRSensorMid)) == HIGH)
                turnleft();
            break;
        }
        startTime = 0;
    }
    if (!sensorLeftHigh || !sensorRightHigh)
    {
        startTime = 0; 
    }
    int temp = nextStop;
    auto it = rotate_cout_with_ids.find(latestStop);
    int currentTime = millis();
    switch (latestStop)
    {

    case 'A':
    case 'B':
    case 'C':
    case 'D':
        if (stoparr[(int)latestStop - 65] == 1)
        {
            stopRoutine();
            stoparr[(int)latestStop - 65] = 0;
        }
        if (latestDirection == 'B')
        {
            if (it == rotate_cout_with_ids.end())
            {
                rotate_cout_with_ids[latestStop] = 1;
                rotate();
            }
            receivingInputs();
            NormalMovement();
        }
        break;
    case 'E':
        if (stoparr[(int)latestStop - 65] == 1)
        {
            stopRoutine();
            stoparr[(int)latestStop - 65] = 0;
        }
        if (nextStop == 'd')
        {

            if (latestDirection == 'R' || latestDirection == 'L')
            {
                if (it == rotate_cout_with_ids.end())
                {
                    rotate_cout_with_ids[latestStop] = 1;
                    rotate();
                }
                receivingInputs();
                NormalMovement();
            }
        }
        break;
    case 'x':
        if (nextStop != 'a')
        {
            while (millis() - currentTime < 1000)
            {
                NormalMovement();
            }
            ending();
            delay(1000000);
        }
        break;
    default:
        break;
    }
}

/**
 * Function Name: loop
 * Input: None
 * Output: None
 * Logic: Continuously listens for UDP packets until one is received.
 *        Once a packet is received, processes inputs and performs movements using the movement() and NormalMovement() functions.
 * Example Call: loop();
 */

void loop()
{
    while (packetreceived != 1)
    {
        receivingInputs();
    }
    receivingInputs();
    movement();
    NormalMovement();
}