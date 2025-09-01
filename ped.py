import cv2

# Open the video file
cap = cv2.VideoCapture("test_video.mp4")  # ✅ Make sure this file exists!

# Check if video opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video file. Check the path or file name.")
    exit()

# Initialize the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("✅ Video started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video or cannot read the frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    regions, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.05)

    # Draw rectangles around detected people
    for i, (x, y, w, h) in enumerate(regions, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {i}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 👀 Show the video frame with detections
    cv2.imshow("Pedestrian Detection", frame)

    # 🛑 Wait for 1ms and check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❎ Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()




