#include <Stepper.h>

#define STEPS 2048  // Number of steps per revolution for 28BYJ-48 motor
Stepper stepper(STEPS, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);
  stepper.setSpeed(10);  // RPM speed
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'L') {      // Move Left
      stepper.step(50);
    }
    else if (command == 'R') { // Move Right
      stepper.step(-50);
    }
  }
}