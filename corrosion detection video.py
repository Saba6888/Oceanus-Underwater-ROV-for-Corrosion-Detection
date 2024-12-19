import numpy as np
import cv2
from dotenv import load_dotenv
import os

load_dotenv()
video_url = os.getenv('VIDEO_URL')

vid = cv2.VideoCapture(video_url)

while True:
    # Read a frame from the video feed
    ret, img = vid.read()
    if not ret:
        print("Failed to capture video. Check the video source.")
        break

    # Resize the frame for consistent processing
    img = cv2.resize(img, (300, 300))

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the color range associated with corrosion
    lower_bound = np.array([0, 50, 50])  # Adjust as needed for specific corrosion colors
    upper_bound = np.array([15, 255, 255])

    # Create a mask to extract regions within the defined color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Binarize the mask to ensure it's a clear black-and-white image
    _, maskbin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of corroded area in the frame
    height, width = maskbin.shape
    total_pixels = height * width
    corrosion_percentage = cv2.countNonZero(maskbin) / float(total_pixels)

    # Output corrosion status
    if corrosion_percentage > 0:
        print(f"Corrosion detected: {corrosion_percentage * 100:.2f}% of the area.")
    else:
        print("No corrosion detected.")

    # Highlight the corroded areas on the original image
    corrosion_detection = cv2.bitwise_and(img, img, mask=mask)

    # Stack the original, HSV, and corrosion detection results for visualization
    results = np.hstack([img, hsv, corrosion_detection])

    # Display the results
    cv2.imshow('Corrosion Detection Results', results)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
