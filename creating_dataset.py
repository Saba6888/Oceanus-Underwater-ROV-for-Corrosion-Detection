import openpyxl
import os
import pandas as pd
import cv2
import numpy as np

def extract_path(directory):
    img_paths = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img_paths.append(img_path)
    return img_paths

def get_corrosion_type(directory):
    corrosion_types = []
    for filename in os.listdir(directory):
        index = filename.index('_')
        corrosion_types.append(filename[0:index])
    return corrosion_types

def get_corrosion_level(directory):
    corrosion_levels_value = []
    corrosion_levels = []
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
        corrosion_levels_value.append(percentage)

        if 0.2 > percentage > 0:
            level = "Light Corrosion"
        elif 0.5 > percentage > 0.2:
            level = "Moderate Corrosion"
        elif 0.8 > percentage > 0.5:
            level = "Severe Corrosion"
        elif 1.0 > percentage > 0.8:
            level = "Critical Corrosion"
        else:
            level = "Negligible Corrosion"

        corrosion_levels.append(level)
    
    return corrosion_levels, corrosion_levels_value

def is_corroded(directory):
    corrosion_status = []
    for filename in os.listdir(directory):
        if 'noCorrosion' in filename:
            corrosion_status.append('no')
        else:
            corrosion_status.append('yes')
    return corrosion_status


if __name__ == "__main__":

    column_titles = {
        'Image Path': [],
        'Type of Corrosion': [],
        'Corrosion Value': [],
        'Corrosion Level': [],
        'Is Corroded': []
    }

    df = pd.DataFrame(column_titles)
    print(df)

    img_paths = extract_path('data')
    corrosion_types = get_corrosion_type('data')
    corrosion_levels, corrosion_levels_value = get_corrosion_level('data')
    status = is_corroded('data')

    for i in range(len(img_paths)):
        df = df.append({'Image Path': img_paths[i], 'Type of Corrosion': corrosion_types[i], 'Corrosion Value': corrosion_levels_value[i], 'Corrosion Level': corrosion_levels[i], 'Is Corroded': status[i]}, ignore_index=True)
    
    df.to_excel('corrosion_data.xlsx', index=False)
    print("File created successfully")