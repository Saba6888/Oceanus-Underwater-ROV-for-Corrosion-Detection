import cv2
import numpy as np

# Read the input image
vid = cv2.VideoCapture('https://192.168.184.148:8080/video')
# vid = cv2.VideoCapture(0)

while (True):
    ret, img = vid.read()
    img = cv2.resize(img, (300, 300))

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the color range associated with corrosion
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([15, 255, 255])

    # Create a mask using the inRange function to extract the corroded regions
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply a morphological operation to improve the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ret, maskbin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    #calculate the percentage
    height, width = maskbin.shape
    size=height * width
    percentage=cv2.countNonZero(maskbin)/float(size)
    if percentage>0:
        print(True)
    else:
        print(False)


    # Bitwise AND operation to get the corroded regions in the original image
    corrosion_detection = cv2.bitwise_and(img, img, mask=mask)

    # Display the original image and the corrosion detection result
    results = np.hstack([img, hsv, corrosion_detection])
    cv2.imshow('results', results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
