#include <WiFi.h>
#include <WiFiUdp.h>
#include <unordered_map>
const char *ssid = "abc";
const char *password = "11111111";
WiFiUDP udp;

unsigned int localUdpPort = 4210; // local port to listen on
char latestStop;
char latestDirection;
char nextStop;
int maskTemp;
int stoparr[5] = {0,0,0,0,0};
// Define pin numbers for various components
const int LEDRed = 13;
const int LEDGreen = 12;
const int Buzzer = 23;
const int MotorClockLeft = 16;
const int MotorClockRight = 4;
const int MotorAntiClockLeft = 2;
const int MotorAntiClockRight = 17;
const int MotorLeftSpeed = 5; // for controlling the speed of left motor
const int MotorRightSpeed = 15; // for controlling the speed of right motor
const int IRSensorFarLeft = 19;
const int IRSensorLeft = 21;
const int IRSensorFarRight = 18;
const int IRSensorRight = 22;
std::unordered_map<int,int> rotate_cout_with_ids; // used in movement function to ensure that the robot only rotates once.
void setup()
{
    digitalWrite(Buzzer, HIGH);
    Serial.begin(115200);
    setupPins();
    setupWiFi();

    // Start listening for UDP packets
    udp.begin(localUdpPort);
    Serial.print("Listening on UDP port ");
    Serial.println(localUdpPort);
}

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

void setupPins()
{
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
}
/**
 * Function: turnRight
 * Description: Initiates a right turn maneuver for the device based on sensor input.
 * Initially, it calls the `right()` function to start turning the device right, followed by a fixed delay of 800 milliseconds to allow the turn to begin.
 * After this delay, it enters a loop, continually calling the `right()` function as long as the left IR sensor (connected to IRSensorLeft) detects 
 * a clear path (signaled by a LOW reading).
 * This method ensures that the device continues turning right until it no longer detects a clear path on the left, indicating the turn has been made.
 * Parameters: None
 * Returns: This function returns nothing.
 */

void turnRight()
{
    right();
    delay(800);
    while (digitalRead(IRSensorLeft) == LOW)
    {
        right();
    }
}
/**
 * Function: turnLeft
 * Description: Executes a left turn maneuver for the device based on sensor input.
 * The process begins with a call to the `left()` function to initiate the left turn. 
 * This is immediately followed by an 800-millisecond delay to establish the turn's commencement. 
 * Subsequently, the function enters a while-loop, continuously executing the `left()` function as long as the right IR sensor (IRSensorRight) indicates a clear path (LOW signal).
 * This loop ensures the device persists in turning left until the sensor no longer detects a clear path on right, indicating the turn has been made.
 * Parameters: None.
 * Returns: This function returns nothing.
 */

void turnLeft()
{
    left();
    delay(800);
    while (digitalRead(IRSensorRight) == LOW)
    {
        left();
    }
}
/**
 * Function: receivingInputs
 * Description: This function is responsible for receiving inputs through UDP packets. 
 * It reads the packet data into a buffer, extracts two integers and a character from the buffer, and then performs operations based on the extracted values.
 * Specifically, it parses the first two sets of numbers as integers, extracts a single character, and adjusts global variables based on these values.
 * It also handles a specific condition based on the value at buffer[7] to update a stops array.
 * Parameters: None. It uses the global `udp` instance to read incoming UDP packets.
 * Returns: This function returns nothing.
 */
void receivingInputs()
{  
    int packetSize = udp.parsePacket();

    if (packetSize)
    {   
        char buffer[512];
        int len = udp.read(buffer, 512);
        if (len > 0)
        {
            buffer[len] = '\0';
        // Converting buffer[0] and buffer[1] to an integer
          int integer1 = atoi(buffer);
          int integer2 = atoi(buffer + 3);

          // The character is directly at buffer[6]
          char character = buffer[6];
          maskTemp=int(buffer[7])-48;
                    // Print the values
          Serial.print("First Integer: ");
          Serial.println(integer1);
          Serial.print("Second Integer: ");
          Serial.println(integer2);
          Serial.print("Character: ");
          Serial.println(character);

          latestStop = integer1;
          nextStop = integer2;
          latestDirection = character;
          if (maskTemp == 1)
          {
            stoparr[nextStop - 95] = 1;
          }
        }
    }
}
/*
 * Function: forward
 * Description: This function activates the motors to move a device forward. 
 * It achieves this by setting the digital pins connected to the motor drivers in a configuration that turns both the left and right motors in the forward direction.
 * Specifically, it sets the pins for clockwise rotation to HIGH and the pins for anti-clockwise rotation to LOW for both left and right motors, 
   effectively enabling forward movement. Additionally, it sets the speed control pins for both motors to HIGH, indicating full speed.
 * Parameters: None. 
 * Returns: This function returns nothing.
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
 * Function: left
 * Description: This function is designed to steer a device to the left by manipulating the motor control pins connected to the device's motors. 
   It sets the digital output to achieve a turning motion by slowing down or stopping the left motor while keeping the right motor active.
 * Specifically, it turns off the left motor by setting its clock direction to LOW and its speed control to LOW, while the right motor is set
   to move forward with its clock direction set to HIGH and speed control also set to HIGH. This differential in motor speeds causes the device to turn left.
 * Parameters: None
 * Returns: This function returns nothing.
 * The logic here is that reducing the speed or stopping one side's motors while the other side's motors are active results in a turn.
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
 * Function: right
 * Description: This function controls the device's motors to steer it to the right. 
 * It achieves this directional movement by setting the digital pins connected to the motor drivers in a way that activates the left motor while slowing down or stopping the right motor. 
 * Specifically, the function sets the left motor to move forward by setting its clockwise rotation pin to HIGH and its speed control to HIGH, while the right motor is effectively stopped or slowed down by setting its clockwise rotation pin to LOW and its speed control to LOW. This configuration creates a differential in motor speeds, causing the device to turn right.
 * Parameters: None.
 * Returns: This function returns nothing.
 * The logic here is that reducing the speed or stopping one side's motors while the other side's motors are active results in a turn.
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
 * Function: buzzerBeep_1s
 * Description: Activates the buzzer for a duration of one second. 
 * This function toggles the state of the digital pin connected to the buzzer, initially setting it to LOW to activate the buzzer, 
    then pausing for 1000 milliseconds (1 second), and finally setting it to HIGH to deactivate the buzzer. 
 * This sequence produces a single beep sound lasting for one second. The function is useful for audible notifications or alerts.
 * Parameters: None
 * Returns: This function returns nothing.
 */

void buzzerBeep_1s()
{
    digitalWrite(Buzzer, LOW);
    delay(1000);
    digitalWrite(Buzzer, HIGH);
}
/**
 * Function: buzzerBeep_5s
 * Description: This function is designed to activate the buzzer for a duration of five seconds.
 * It achieves this by setting the digital pin connected to the buzzer to LOW, thereby turning the buzzer on, 
   followed by a delay of 5000 milliseconds (5 seconds), and then setting the pin to HIGH to turn the buzzer off.
 * This operation results in a beep sound that lasts for five seconds, suitable for longer notifications or alerts where a more prolonged signal is desired.
 * Parameters: None. 
 * Returns: This function returns nothing.
*/

void buzzerBeep_5s()
{
    digitalWrite(Buzzer, LOW);
    delay(5000);
    digitalWrite(Buzzer, HIGH);
}
/**
 * Function: buzzerBeep_1s_and_Led
 * Description: Simultaneously activates a LED and a buzzer for a duration of one second. 
 * This function is designed to provide both visual and audible alerts.
 * It starts by turning the LED (connected to the LEDRed pin) on and activating the buzzer (connected to the Buzzer pin) by setting the LEDRed pin to HIGH
   and the Buzzer pin to LOW. After a delay of 1000 milliseconds (1 second), it turns both the LED and the buzzer off by setting the LEDRed pin to LOW
   and the Buzzer pin to HIGH. This sequence creates a synchronized light and sound signal lasting for one second.
 * Parameters: None
 * Returns: This function returns nothing.
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
 * Function: stop
 * Description: This function stops all motor activity by setting the digital output pins associated with both the left and right motors to LOW.
 * It effectively deactivates the motors by turning off both the clockwise and anti-clockwise rotation signals as well as the speed control signals for 
   each motor. This operation ensures the device ceases movement, allowing for a controlled stop.
 * Parameters: None
   MotorAntiClockRight, MotorLeftSpeed, and MotorRightSpeed.
 * Returns: This function returns nothing as it is a void type.
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
 * Function: rotate
 * Description: Initiates a rotation maneuver by setting the motors to rotate the device in a specific direction until a certain condition is met,
   detected by an infrared (IR) sensor.
  The function starts by printing a message to the serial monitor indicating that a rotation is occurring.
   It then sets the motor control pins to initiate rotation: the left motor is set to rotate clockwise and the right motor counter-clockwise,
    both at high speed, to achieve a pivoting motion. 
   After a delay of 1150 milliseconds, which allows the device to start the rotation, it enters a loop where it continues to execute the 'right()' function 
   (presumably to keep rotating right) until the IR sensor on the far right detects an object (or a specific condition is met, indicated by the sensor's 
   output going HIGH). Once this condition is met, the 'stop()' function is called to halt all motor activity, thus completing the rotation maneuver.
 * Parameters: None 
   MotorLeftSpeed, MotorRightSpeed) and sensor input (IRSensorFarRight).
 * Returns: This function returns nothing.
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
    delay(1150);
    while(digitalRead(IRSensorFarRight) != HIGH)
    {
      right();
    }
    stop();
}
/**
 * Function: stopRoutine
 * Description: Executes a routine that first brings the device to a complete stop and then issues a one-second beep using the buzzer.
   This function is a higher-level abstraction combining two primary actions: stopping all motor activity and providing an audible alert. 
   It is designed to be called in scenarios where an immediate halt of the device's motion is required, followed by an alert signal, 
   potentially to indicate the completion of an operation or to warn of an emergency stop.
 * Parameters: None
   to activate the buzzer for one second.
 * Returns: This function returns nothing.
 */

void stopRoutine()
{
    stop();
    buzzerBeep_1s();

}
/**
 * Function: ending
 * Description: This function signifies the completion of a process or routine by executing a sequence of actions: it first commands the device to halt any motion by calling the `stop()` function,
  and then activates the buzzer for a five-second duration through the `buzzerBeep_5s()` function. The extended beep is intended to provide a clear and noticeable signal that an operation has concluded, suitable for situations where a longer alert is preferred to ensure the ending is distinctly recognized.
 * Parameters: None
 * Returns: This function returns nothing.
 */

void ending()
{
    stop();
    buzzerBeep_5s();
}
/**
 * Function: NormalMovement
 * Description: Controls the device's movement based on infrared (IR) sensor readings, enabling it to navigate by responding to sensor inputs. 
   The function reads data from four IR sensors positioned to detect obstacles or navigational cues from different directions.
   Then it decides whether to move forward, turn left, or turn right. Specifically, it initiates a left turn
   if the left sensors detect a navigational cue (e.g., an obstacle or line on the left side), a right turn if the right sensors detect a cue, and moves forward if no specific cues are detected that warrant a turn.
 * Parameters: None. 
 * Returns: This function returns nothing as it is a void type.
 */

void NormalMovement()
{
    int leftSensorData = digitalRead(IRSensorLeft);
    int rightSensorData = digitalRead(IRSensorRight);
    int farleftSensorData = digitalRead(IRSensorFarLeft);
    int farrightSensorData = digitalRead(IRSensorFarRight);

    if (leftSensorData == HIGH && rightSensorData == LOW || farleftSensorData == LOW && farrightSensorData == HIGH)
    {
        left();
    }
    else if (leftSensorData == LOW && rightSensorData == HIGH || farleftSensorData == HIGH && farrightSensorData == LOW)
    {

        right();
    }
    else
    {
        forward();
    }
}
/**
 * Function: movement
 * Description: This function orchestrates the device's movement based on the current direction indicated by the global variable `latestDirection` and
  the status of infrared (IR) sensors. It selects an action (forward, turn right, or turn left) depending on `latestDirection`. 
  Actions are conditional upon both the left and right IR sensors detecting a specific condition (e.g., HIGH signal indicating a path is clear). 
  Additionally, the function handles specific stop conditions based on the value of `latestStop`, executing routines like `stopRoutine()`, `rotate()`, 
  `receivingInputs()`, `NormalMovement()`, or `ending()` depending on various cases, including predefined stop positions and special actions when the 
  direction is 'B' (backwards) or when a termination condition is met.
 * Parameters: None
    states directly using `digitalRead()`.
 * Returns: This function returns nothing.
 */

void movement()
{   
    switch(latestDirection)
    {
        case 'F':
            if((digitalRead(IRSensorLeft) && digitalRead(IRSensorRight)) == HIGH)
                forward();
            break;
        case 'R':
            if((digitalRead(IRSensorLeft) && digitalRead(IRSensorRight)) == HIGH)
                turnRight();
            break;
        case 'L':
            if((digitalRead(IRSensorLeft) && digitalRead(IRSensorRight)) == HIGH)
                turnLeft();
            break;

    }

    int temp = nextStop;
    auto it = rotate_cout_with_ids.find(latestStop);
    switch(latestStop)
    {
        case 95: case 96: case 97: case 98: case 99: 
            if(stoparr[latestStop-95]==1){
            stopRoutine();
            stoparr[latestStop-95]=0;
            }
            if(latestDirection == 'B')
            {
                if(it == rotate_cout_with_ids.end()) {
                    rotate_cout_with_ids[latestStop] = 1;
                    rotate();
                }
                receivingInputs();
                NormalMovement();
            }
            break;
        case 23:
            ending();
            delay(1000000);
            break;
        default:
            break;
    }

}

/**
 * Function: loop
 * Description: Serves as the main operational cycle of the device, continuously executing three key functions to drive its behavior. 
 * This function is the core of the device's runtime logic, called repeatedly to ensure responsive and continuous operation. 
 * It starts by calling `receivingInputs()` to gather and process incoming data, likely from sensors or network communication.
 * Following this, it executes `movement()`, which decides the device's next action based on the latest inputs and internal logic. 
 * Finally, it calls `NormalMovement()` to adjust the device's movement based on predefined conditions or sensor feedback.
 * Together, these functions allow the device to respond dynamically to its environment and internal state changes.
 * Parameters: None.
 * Returns: This function returns nothing.
 */

void loop()
{
    receivingInputs();
    movement();
    NormalMovement();

}