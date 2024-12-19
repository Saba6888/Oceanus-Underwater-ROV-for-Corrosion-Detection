import cv2
import numpy as np
import os

# Read the input image
for img in os.listdir('data'):
    img_path = os.path.join('data', img)
    img = cv2.imread(img_path)

    img = cv2.resize(img, (300,300))


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
    print(percentage, end="    ->    ")
    if 0.2 > percentage > 0.003:
        print("Light Corrosion")
    elif 0.5 > percentage > 0.2:
        print("Moderate Corrosion")
    elif 0.8 > percentage > 0.5:
        print("Severe Corrosion")
    elif 1.0 > percentage > 0.8:
        print("Critical Corrosion")
    else:
        print("Negligible Corrosion")
    # Bitwise AND operation to get the corroded regions in the original image
    corrosion_detection = cv2.bitwise_and(img, img, mask=mask)

    # Display the original image and the corrosion detection result
    # cv2.imshow('Original Image', img)
    # cv2.imshow('HSV Image', hsv)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Corrosion Detection', corrosion_detection)
    results = np.hstack([img, hsv, corrosion_detection])
    # cv2.imshow('results', results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
