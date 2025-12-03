

#include <Bluepad32.h>

// =====================================================================
// 1. Motor Pin Definitions, Constants
// =====================================================================

// For robot id 4 flips right pins due to hardware soldering error
// #define LEFT_DIR_FORWARD 23
// #define LEFT_DIR_BACKWARD 21
// #define RIGHT_DIR_FORWARD 25
// #define RIGHT_DIR_BACKWARD 33

// For robots id 1-3
#define LEFT_DIR_FORWARD    23
#define LEFT_DIR_BACKWARD   21
#define RIGHT_DIR_FORWARD   33
#define RIGHT_DIR_BACKWARD  25

#define LEFT_PWM 19
#define RIGHT_PWM 32

// According to the analogWrite it can only take values 0-255
const uint8_t MAX_SPEED = 255;

// We’ll keep the same “PS4Controller” style logic by mapping the
// Bluepad32 joystick range (-511..512) to approx. -128..127.
// Increase if stick is fluctuating over these values while at rest
const int STICK_CENTER_MIN = -10; // Lower bound for center deadzone
const int STICK_CENTER_MAX = 10;  // Upper bound for center deadzone

// How hard you must press L2 / R2 (0..1023 in Bluepad32)
const int TRIGGER_THRESHOLD = 200;

// Global pointer to our *single* gamepad
ControllerPtr myGamepad = nullptr;

// =====================================================================
// 2. Motor Control Functions
// =====================================================================

// 2.1 Left Motor Control Functions
// ---------------------------------------------------------------------
void spinLeftMotor(int16_t speed) {
  if (speed > 0) {
    
    // Forward
    digitalWrite(LEFT_DIR_FORWARD, HIGH);
    digitalWrite(LEFT_DIR_BACKWARD, LOW);
    analogWrite(LEFT_PWM, speed);

  } 
  
  else if (speed < 0) {
    
    // Backward
    digitalWrite(LEFT_DIR_FORWARD, LOW);
    digitalWrite(LEFT_DIR_BACKWARD, HIGH);
    analogWrite(LEFT_PWM, -speed);

  } 
  
  else {
    // Stop
    digitalWrite(LEFT_DIR_FORWARD, LOW);
    digitalWrite(LEFT_DIR_BACKWARD, LOW);
    analogWrite(LEFT_PWM, 0);
  }
}

// 2.2 Right Motor Control Functions
// ---------------------------------------------------------------------
void spinRightMotor(int16_t speed) {
  if (speed > 0) {
    // Forward
    digitalWrite(RIGHT_DIR_FORWARD, HIGH);
    digitalWrite(RIGHT_DIR_BACKWARD, LOW);
    analogWrite(RIGHT_PWM, speed);
  } else if (speed < 0) {
    // Backward
    digitalWrite(RIGHT_DIR_FORWARD, LOW);
    digitalWrite(RIGHT_DIR_BACKWARD, HIGH);
    analogWrite(RIGHT_PWM, -speed);
  } else {
    // Stop
    digitalWrite(RIGHT_DIR_FORWARD, LOW);
    digitalWrite(RIGHT_DIR_BACKWARD, LOW);
    analogWrite(RIGHT_PWM, 0);
  }
}

// 2.3 Left and Right Motors Stop Function
// ---------------------------------------------------------------------
void stopMotors() {
  spinLeftMotor(0);
  spinRightMotor(0);
}

// =====================================================================
// 3. Bluepad32 Gamepad Handling
// =====================================================================

// 3.1 Connection and Disconnection Event Handlers
// ---------------------------------------------------------------------
void onConnectedGamepad(ControllerPtr gp) {
  // Use the first connected controller as the driver
  if (myGamepad == nullptr) {
    myGamepad = gp;
    Serial.printf("Gamepad connected: index=%d, model=%s\n",
                  gp->index(), gp->getModelName());
    // Optional: turn on player LED 1
    gp->setPlayerLEDs(0x1);
  } else {
    Serial.println("Another controller connected, but one is already in use.");
  }
}

void onDisconnectedGamepad(ControllerPtr gp) {
  if (gp == myGamepad) {
    Serial.println("Gamepad disconnected");
    myGamepad = nullptr;
    stopMotors();
  }
}

// 3.2 Read gamepad and drive motors (equivalent of onPS4Notify)
// ---------------------------------------------------------------------
void handleGamepad(ControllerPtr ctl) {
  if (!ctl || !ctl->isConnected()) {
    stopMotors();
    return;
  }

  // Bluepad32 ranges:
  //   axisX: -511 .. 512  (left stick X) :contentReference[oaicite:2]{index=2}
  //   brake():   0 .. 1023  (L2)
  //   throttle():0 .. 1023  (R2)
  //
  // Map axisX to approx. -128..127 and invert like your original code:
  int left_stick_x = -(ctl->axisX() / 4);  // keep same sign convention

  bool stick_outside_deadzone =
      (left_stick_x < STICK_CENTER_MIN || left_stick_x > STICK_CENTER_MAX);

  // Triggers as digital-style "pressed" flags
  bool l2_pressed = (ctl->brake()    > TRIGGER_THRESHOLD);  // L2 = brake()
  bool r2_pressed = (ctl->throttle() > TRIGGER_THRESHOLD);  // R2 = throttle()


  // --------------------------
  // 3.2.1 FORWARD (R2 only)
  // --------------------------
  if (r2_pressed && !l2_pressed) {
    
    // Proportional speed control
    int forward_speed = MAX_SPEED;

    int16_t left_speed  = forward_speed;
    int16_t right_speed = forward_speed;

    if (stick_outside_deadzone) {
      
      // Steer left: reverse left motor speed, keep right speed max forward
      // Steer left: anti-clockwise circle motion to the left of the robot
      if (left_stick_x < STICK_CENTER_MIN) {
        
        // Normalize steering: 0 = none , 0.5 = half, 1 = full
        float steer_degree = float(abs(left_stick_x)) / 128.0f;
        
        // Convert steering to speed
        // steer_degree = 0       -> +255     forward
        // steer_degree = ]0,0.5[ -> 255-0    slight curve
        // steer_degree = 0.5     -> 0        pivot turn
        // steer_degree = ]0.5,1[ -> 0-(-255) harder turn
        // steer_degree = 1       -> -255     hardest turn
        left_speed = int16_t(forward_speed * (1.0f - 2.0f * steer_degree));

      }

      // Steer right: reverse right motor speed, keep left motorspeed max forward
      // Steer right: clockwise circle motion to the right of the robot
      else if (left_stick_x > STICK_CENTER_MAX) {
        
        // Normalize steering: 0 = none , 0.5 = half, 1 = full
        float steer_degree = float(abs(left_stick_x)) / 127.0f;
        
        // Convert steering to speed
        // steer_degree = 0       -> +255     forward
        // steer_degree = ]0,0.5[ -> 255-0    slight curve
        // steer_degree = 0.5     -> 0        pivot turn
        // steer_degree = ]0.5,1[ -> 0-(-255) harder turn
        // steer_degree = 1       -> -255     hardest turn
        right_speed = int16_t(forward_speed * (1.0f - 2.0f * steer_degree));

      }

      // Putting the motor speed values the previous if or else if statement 
      // defined into action
      spinLeftMotor(left_speed);
      spinRightMotor(right_speed);

    } 
    
    else {
      // Straight forward
      spinLeftMotor(forward_speed);
      spinRightMotor(forward_speed);
    }
  }


  // --------------------------
  // 3.2.2 BACKWARD (L2 only)
  // --------------------------
  else if (l2_pressed && !r2_pressed) {
    
    // Proportional speed control
    int backward_speed = -MAX_SPEED;

    int16_t left_speed  = backward_speed;
    int16_t right_speed = backward_speed;

    if (stick_outside_deadzone) {
      

      // Steer to the left: reverse left motor speed
      // causes robot to go backward to the left clockwise circle motion
      if (left_stick_x < STICK_CENTER_MIN) {
        
         // Normalize steering: 0 = none , 0.5 = half, 1 = full
        float steer_degree = float(abs(left_stick_x)) / 128.0f;
        
        // Convert steering to speed
        // steer_degree = 0       -> -255     backward
        // steer_degree = ]0,0.5[ -> -255-0    slight curve
        // steer_degree = 0.5     -> 0        pivot turn
        // steer_degree = ]0.5,1[ -> 0-255 harder turn
        // steer_degree = 1       -> +255     hardest turn
        left_speed = int16_t(backward_speed * (1.0f - 2.0f * steer_degree));

      }

      // Steer to the right while going backwards: reverse right motor speed
      // causes robot to go backwards to the right anti-clockwise circle motion
      else if (left_stick_x > STICK_CENTER_MAX) {
        
        // Normalize steering: 0 = none , 0.5 = half, 1 = full
        float steer_degree = float(abs(left_stick_x)) / 127.0f;
        
        // Convert steering to speed
        // steer_degree = 0       -> -255     forward
        // steer_degree = ]0,0.5[ -> -255-0    slight curve
        // steer_degree = 0.5     -> 0        pivot turn
        // steer_degree = ]0.5,1[ -> 0-255 harder turn
        // steer_degree = 1       -> +255     hardest turn
        right_speed = int16_t(backward_speed * (1.0f - 2.0f * steer_degree));

      }

      // Collect the results of the previous logics and set speeds
      spinLeftMotor(left_speed);
      spinRightMotor(right_speed);

    } 
    
    else {
      // Straight backward
      spinLeftMotor(backward_speed);
      spinRightMotor(backward_speed);
    }
  }


  // --------------------------
  // 3.2.3 STOP (no trigger / both?)
  // --------------------------
  else {
    stopMotors();
  }
}


// =====================================================================
// 4. Setup
// =====================================================================


void setup() {
  Serial.begin(115200);
  delay(200);

  // Motor pins
  pinMode(LEFT_DIR_BACKWARD, OUTPUT);
  pinMode(LEFT_DIR_FORWARD,  OUTPUT);
  pinMode(LEFT_PWM,          OUTPUT);
  pinMode(RIGHT_DIR_BACKWARD,OUTPUT);
  pinMode(RIGHT_DIR_FORWARD, OUTPUT);
  pinMode(RIGHT_PWM,         OUTPUT);

  stopMotors();

  // Bluepad32 init
  BP32.setup(&onConnectedGamepad, &onDisconnectedGamepad);
  //BP32.enableNewBluetoothConnections(true);

  Serial.println("Bluepad32 robot ready. Pair PS4: HOLD PS + SHARE until light flashes.");
}


// =====================================================================
// 5. Loop
// =====================================================================


void loop() {
  // Poll Bluepad32 (must be called frequently)
  BP32.update();

  if (myGamepad && myGamepad->isConnected()) {
    handleGamepad(myGamepad);
  } else {
    // Safety: no controller → stop
    stopMotors();
  }

  // Small delay to avoid hammering CPU
  delay(10);
}
