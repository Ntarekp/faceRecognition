import cv2
import time
import math

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(1)

prev_cx, prev_cy = None, None
prev_time = time.time()
direction = ""
speed = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute center of face
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Compare with previous frame
        if prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy

            # Time difference between frames
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            # Speed in pixels per second
            dist = math.sqrt(dx ** 2 + dy ** 2)
            speed = dist / dt if dt > 0 else 0

            # Determine direction
            if abs(dx) > abs(dy):
                direction = "Right" if dx > 10 else "Left" if dx < -10 else "Stationary"
            else:
                direction = "Down" if dy > 10 else "Up" if dy < -10 else "Stationary"

        # Update previous center
        prev_cx, prev_cy = cx, cy

        # Display direction and speed
        cv2.putText(frame, f"Direction: {direction}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed:.2f} px/s", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the result
    cv2.imshow('Face Direction Tracker', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
