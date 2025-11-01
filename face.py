import cv2
import time
import serial

# ========== CONFIGURATION ==========
SERIAL_PORT = 'COM23'
BAUD_RATE = 9600
CAMERA_INDEX = 1
MOTOR_STEP = 3          # Simulated degrees per move
MIN_POS = 0
MAX_POS = 180

# ==================================
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # Wait for Arduino to initialize

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

prev_cx, prev_cy = None, None
prev_time = time.time()

# Track motor position locally (simulated)
motor_angle = 90  # start centered
direction = "Stationary"

print("\n🎯 FACE TRACKER PRESENTATION MODE")
print("Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    current_time = time.time()
    time_diff = current_time - prev_time if prev_time else 1.0

    direction = "Stationary"
    speed = 0.0

    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        if prev_cx is not None:
            dx = cx - prev_cx
            distance = abs(dx)

            if distance > 15:  # movement threshold
                direction = "Left" if dx > 0 else "Right"
                speed = distance / time_diff if time_diff > 0 else 0

                # Send to Arduino
                if direction == "Left":
                    arduino.write(b'L')
                    motor_angle = min(MAX_POS, motor_angle + MOTOR_STEP)
                elif direction == "Right":
                    arduino.write(b'R')
                    motor_angle = max(MIN_POS, motor_angle - MOTOR_STEP)

                print(f"Direction: {direction}, Speed: {speed:.2f}, Motor Angle: {motor_angle}°")
            else:
                direction = "Stationary"
        else:
            direction = "Stationary"

        prev_cx, prev_cy = cx, cy
        prev_time = current_time
        break  # process only the first face

    # ===== Overlay UI =====
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Add transparent bar at bottom
    cv2.rectangle(overlay, (0, h - 100), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Status text
    cv2.putText(frame, f"Direction: {direction}", (20, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Motor Position: {motor_angle}°", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Speed: {speed:.1f} px/s", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show camera feed
    cv2.imshow('Face Tracker - Presentation Mode', frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
arduino.close()
cv2.destroyAllWindows()
