import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model("corrosion_classification_model.h5")  # Replace "your_model_path" with the path to your saved model

# Create a function to preprocess the input frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture('https://100.86.101.71:8080/video')
# cap = cv2.VideoCapture(1)


while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Resize the image
    frame = cv2.resize(frame, (300,300))

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Make predictions
    prediction = model.predict(processed_frame)

    # Display the result on the frame
    if prediction > 0.5:
        cv2.putText(frame, "Corroded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Not Corroded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Display the frame
    output = np.hstack([frame, hsv])
    cv2.imshow("Live Feed", output)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
