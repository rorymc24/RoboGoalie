# The following is a college project python script for the robo goalie project.




```
import threading
import numpy as np
import cv2
from collections import deque
from sklearn.linear_model import LinearRegression
import serial  # For communication with Arduino
import time

# Store object positions for trajectory prediction
object_positions = deque(maxlen=10)  # Keep the last 10 positions
last_seen_time = time.time()

# Global variables for tolerance-based prediction update
last_predicted_value = None
last_moving_position = None
TOLERANCE = 12  # Tolerance (in pixels) for ball movement

# Blue line control
blue_line_y = 500  # 150 pixels above the red line
freeze_prediction = False
frozen_prediction = None

# Initialize serial communication with Arduino (ensure correct COM port)
arduino = serial.Serial('COM5', 9600, timeout=1)  # Change 'COM5' to your Arduino port


def serial_monitor():
    print("Python Serial Monitor started. Press Ctrl+C to exit.\n")
    try:
        while True:
            if arduino.in_waiting > 0:
                data = arduino.readline().decode().strip()
                if data:
                    print(f"Arduino says: {data}")
    except KeyboardInterrupt:
        print("Exiting Serial Monitor.")


def cam_int():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError("Could not open webcam.")
    return cam


def predict_trajectory(positions, target_y, frame_width):
    if len(positions) < 2:
        return None

    x_coords = np.array([pos[0] for pos in positions]).reshape(-1, 1)
    y_coords = np.array([pos[1] for pos in positions])

    model = LinearRegression()
    model.fit(y_coords.reshape(-1, 1), x_coords)
    m = model.coef_[0].item()
    c = model.intercept_.item()

    predicted_x = max(0, min(int(m * target_y + c), frame_width))
    return (predicted_x, target_y), m, c


def map_to_range(predicted_x, screen_width):
    if predicted_x < 0 or predicted_x > screen_width:
        return 100
    mapped_value = (predicted_x / screen_width) * 11
    return max(0, min(int(mapped_value), 11))


def cam_frame(cam, target_y):
    global last_seen_time, last_predicted_value, last_moving_position
    global freeze_prediction, frozen_prediction

    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to read frame from webcam.")

    # Flip the frame horizontally to mirror the view
    frame = cv2.flip(frame, 1)

    original_height, original_width = frame.shape[:2]

    # Resize the height to 1.5x while keeping the aspect ratio
    new_height = int(original_height * 1.5)
    scale_factor = new_height / original_height
    new_width = int(original_width * scale_factor)
    frame = cv2.resize(frame, (new_width, new_height))

    # Crop the width to focus on the center portion
    crop_width = int(original_width * 0.8)  # Adjust the width crop factor (0.8 means 80% of original width)
    start_x = (new_width - crop_width) // 2
    end_x = start_x + crop_width

    # Ensure the crop does not go out of bounds
    frame = frame[:, start_x:end_x]
    frame_width = frame.shape[1]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([5, 150, 150])
    high = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    normalview = frame.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_detected = False
    new_movement = False

    largest_contour = max(contours, key=cv2.contourArea, default=None)
    if largest_contour is not None and cv2.contourArea(largest_contour) > 500:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx, cy = x + w // 2, y + h // 2

        if cy > blue_line_y and cy < target_y:  # Object is between the blue and red lines
            freeze_prediction = True  # Freeze prediction within this region
            frozen_prediction = last_predicted_value

        elif cy < blue_line_y or cy > target_y:  # Object is outside the region
            freeze_prediction = False  # Allow prediction updates outside the region

        if last_moving_position is None or np.linalg.norm(np.array((cx, cy)) - np.array(last_moving_position)) > TOLERANCE:
            object_positions.append((cx, cy))
            last_moving_position = (cx, cy)
            new_movement = True

        last_seen_time = time.time()
        object_detected = True
        cv2.circle(result, (cx, cy), 10, (255, 0, 0), -1)

    cv2.line(result, (0, target_y), (frame_width, target_y), (0, 0, 255), 2)  # Red line
    cv2.line(result, (0, blue_line_y), (frame_width, blue_line_y), (255, 0, 0), 2)  # Blue line

    predicted_value = None
    if object_detected:
        if not freeze_prediction:  # Only update if not frozen
            if new_movement and len(object_positions) >= 2:
                prediction = predict_trajectory(object_positions, target_y, frame_width)
                predicted_point = prediction[0] if prediction else (last_moving_position[0], target_y)
            else:
                predicted_point = (last_moving_position[0], target_y)
            predicted_value = map_to_range(predicted_point[0], frame_width)
            last_predicted_value = predicted_value
            cv2.circle(result, predicted_point, 10, (0, 255, 255), 2)
        else:
            predicted_value = frozen_prediction  # Use frozen prediction

    if time.time() - last_seen_time > 3:
        predicted_value = 100

    display_predicted_value(result, predicted_value)
    return result, predicted_value, normalview


def display_predicted_value(frame, predicted_x):
    text = f"Predicted X: {predicted_x}" if predicted_x is not None else "Prediction unavailable"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    cam = cam_int()
    target_y = 700
    last_sent_time = time.time()
    last_sent_position = None  # Store the last sent position

    try:
        while True:
            result, predicted_value, normalview = cam_frame(cam, target_y)
            cv2.imshow('Filtered Frame', result)
            cv2.imshow('Normal View', normalview)

            # Only send if the predicted value has changed and 0.5 seconds have passed
            if (predicted_value is not None and
                predicted_value != last_sent_position and
                (time.time() - last_sent_time) > 0.5):

                arduino.write(f"{predicted_value}\n".encode())
                last_sent_time = time.time()
                last_sent_position = predicted_value  # Update the last sent position

            if cv2.waitKey(10) == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        arduino.close()


if __name__ == "__main__":
    monitor_thread = threading.Thread(target=serial_monitor, daemon=True)
    monitor_thread.start()
    main()
```



# The following is the corresponding ardunio script for the project




```
#include <AccelStepper.h>

// ----- Configuration -----
#define STEP_PIN               11
#define DIR_PIN                9
#define FORWARD_LIMIT_SWITCH   7
#define BACKWARD_LIMIT_SWITCH  6
#define ENABLE_SWITCH          2  

const int NUM_REGIONS = 11;       // 11 total regions (0-10)
const long CALIBRATION_SPEED = 1000; // Calibration speed
const long RUN_SPEED = 15000;      // Running speed

// ----- Globals -----
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
long regionSteps[NUM_REGIONS]; // Step positions for each region
long maxSteps = 0;
long currentPosition = 0;      // Software position tracking

// Volatile flag used by the ISR to indicate system state changes
volatile bool systemEnabled = false;
volatile bool stateChanged = false;

void setup() {
  pinMode(FORWARD_LIMIT_SWITCH, INPUT_PULLUP);
  pinMode(BACKWARD_LIMIT_SWITCH, INPUT_PULLUP);
  pinMode(ENABLE_SWITCH, INPUT_PULLUP); // Use INPUT_PULLUP if wiring so that LOW indicates closed

  Serial.begin(9600);

  // Setup stepper parameters
  stepper.setMaxSpeed(RUN_SPEED);
  stepper.setAcceleration(15000); // Tune as needed
  delay(1000);

  // Attach an interrupt to the enable switch pin
  attachInterrupt(digitalPinToInterrupt(ENABLE_SWITCH), handleSwitch, CHANGE);
}

void loop() {
  // If the switch state changed, handle it in the main loop
  if (stateChanged) {
    // Disable interrupts temporarily while reading the volatile flag
    noInterrupts();
    bool enabled = systemEnabled;
    stateChanged = false;
    interrupts();

    if (enabled) {
      Serial.println("Switch enabled: Starting calibration.");
      calibrateMotor();
    } else {
      stepper.stop();
      Serial.println("Switch disabled: System paused.");
    }
  }

  // If the system is enabled, run motor commands and check serial input
  if (systemEnabled) {
    // Run the stepper to its target
    stepper.runToPosition();

    // Process serial commands if available
    if (Serial.available() > 0) {
      String value = Serial.readStringUntil('\n');
      int targetRegion = value.toInt();
      Serial.print("Target Region: ");
      Serial.println(targetRegion);

      if (targetRegion == 100) {
        Serial.println("Out of bounds -> Move to center (region 5).");
        moveToRegion(5);
      } else if (targetRegion >= 0 && targetRegion < NUM_REGIONS) {
        moveToRegion(targetRegion);
      } else {
        Serial.println("Invalid region index received.");
      }
    }

    // Optionally check limit switches continuously
    checkLimitSwitches();
  }
}

// Interrupt service routine: update systemEnabled flag
void handleSwitch() {
  systemEnabled = digitalRead(ENABLE_SWITCH);
  stateChanged = true;
}

void calibrateMotor() {
  Serial.println("Starting calibration...");

  // 1) Move forward until the forward limit switch is triggered
  while (digitalRead(FORWARD_LIMIT_SWITCH) == HIGH) {
    stepper.setSpeed(CALIBRATION_SPEED);
    stepper.runSpeed();
  }
  Serial.println("Forward limit switch triggered");
  delay(500);

  // 2) Move backward until the backward limit switch is triggered
  stepper.setCurrentPosition(0);
  while (digitalRead(BACKWARD_LIMIT_SWITCH) == HIGH) {
    stepper.setSpeed(-CALIBRATION_SPEED);
    stepper.runSpeed();
  }
  Serial.println("Backward limit switch triggered");

  // Use the absolute value of the stepper's position as the total steps
  maxSteps = labs(stepper.currentPosition());
  Serial.print("Total steps found: ");
  Serial.println(maxSteps);

  // 3) Calculate region boundaries
  long regionStepSize = maxSteps / (NUM_REGIONS - 1);
  for (int i = 1; i <= NUM_REGIONS; i++) {
    regionSteps[i] = i * regionStepSize;
  }

  Serial.println("Regions:");
  for (int i = 1; i <= NUM_REGIONS; i++) {
    Serial.print("Region ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(regionSteps[i]);
  }
  delay(500);

  // 4) Move to the center region (region 5)
  stepper.setCurrentPosition(11);
  delay(500);
  moveToRegion(5);

  Serial.println("Calibration complete.");
  delay(500);
}

void moveToRegion(int region) {
  if (region < 0 || region >= NUM_REGIONS) {
    Serial.println("Error: Requested out-of-bounds region.");
    return;
  }
  long targetSteps = regionSteps[region];
  Serial.print("Moving from steps ");
  Serial.print(stepper.currentPosition());
  Serial.print(" to ");
  Serial.println(targetSteps);

  stepper.moveTo(targetSteps);
  currentPosition = targetSteps;
}

// Optional: continuous limit switch checking
void checkLimitSwitches() {
  if (stepper.speed() > 0 && digitalRead(FORWARD_LIMIT_SWITCH) == LOW) {
    Serial.println("Forward limit reached! Stopping.");
    stepper.stop();
  }
  if (stepper.speed() < 0 && digitalRead(BACKWARD_LIMIT_SWITCH) == LOW) {
    Serial.println("Backward limit reached! Stopping.");
    stepper.stop();
  }
}

```
