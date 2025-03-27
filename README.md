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
