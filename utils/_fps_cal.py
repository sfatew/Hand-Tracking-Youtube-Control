import cv2
import time

def main():
    # Open the default webcam (0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Failed to grab frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        if curr_time - prev_time > 0.5: # only update FPS every 0.5 second
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0
        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam FPS", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
