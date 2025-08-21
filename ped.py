import cv2

# Load the image
image = cv2.imread('person.jpg')

# Load Haar Cascade for full body detection
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect humans
humans = human_cascade.detectMultiScale(gray, 1.1, 3)

# Draw rectangles around detected humans
for (x, y, w, h) in humans:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the result
cv2.imshow('Detected Humans', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
