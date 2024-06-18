import cv2
import numpy as np
import pytesseract
import re

# face and nose detection xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# video for both cameras
regular_cam = cv2.VideoCapture(0)  # Regular camera
thermal_cam = cv2.VideoCapture(1)  # Thermal camera

def min_and_max(frame_image): 
    # Ensure Tesseract's executable is in your PATH or provide the path directly
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Load the image
    image = cv2.imread(frame_image)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to preprocess the image
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Optional: Apply dilation to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(binary_image, kernel, iterations=1)

    try:
        # Use Tesseract to extract text
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(dilated, config=custom_config)
        # Define patterns for "max" and "min" with associated numbers
        max_pattern = r'\bMax\b\D*(\d+(\.\d+)?)'
        min_pattern = r'\bMin\b\D*(\d+(\.\d+)?)'
        # Find all occurrences of patterns in the text
        max_results = re.findall(max_pattern, text, flags=re.IGNORECASE)
        min_results = re.findall(min_pattern, text, flags=re.IGNORECASE)
        # Extract max and min values
        max_value = float(max_results[0][0]) if max_results else None
        min_value = float(min_results[0][0]) if min_results else None
        return max_value, min_value 
    
    except Exception as e:
        print(f"Error: An exception occurred during OCR: {e}")
        return None, None
    
# Function to convert pixel value to temperature (temperature is in celcius) 
def pixel_to_temperature(pixel_value, min_temp=20, max_temp=100):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)

while True:
    # Capture frame-by-frame from both cameras
    ret1, regular_frame = regular_cam.read()
    ret2, thermal_frame = thermal_cam.read()

    #make sure that ret1 and ret2 are reading properly
    if not ret1 or not ret2:
        print("Failed to grab frames")
        break
    
    max_temp, min_temp = min_and_max(thermal_frame)
    if max_temp is None or min_temp is None:
        print("Failed to extract temperature range")
        continue

    # Convert the regular frame to grayscale
    gray_frame = cv2.cvtColor(regular_frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

   #for coordinates in the faces detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face in the regular frame
        # cv2.rectangle(regular_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]

        # Detect nose within the face ROI
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (nx, ny, nw, nh) in noses:
            nose_center_x = x + nx + nw // 2
            nose_center_y = y + ny + nh // 2
            cv2.rectangle(regular_frame, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (0, 255, 0), 2)
            cv2.circle(regular_frame, (nose_center_x, nose_center_y), 5, (0, 0, 255), -1)

            # Ensure thermal_frame is in single-channel (grayscale)
            if len(thermal_frame.shape) == 3 and thermal_frame.shape[2] == 3:
                thermal_frame_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
            else:
                thermal_frame_gray = thermal_frame

            # if both cameras are on the same plane
            nose_temperature_pixel_value = thermal_frame_gray[nose_center_y, nose_center_x]
            nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=min_temp, max_temp=max_temp)
            print('Temperature at nose:', nose_temperature)

            # Display the temperature on the regular frame
            # cv2.putText(regular_frame, f'{nose_temperature:.2f} °C', (nose_center_x, nose_center_y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(regular_frame, "face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # show the frames of the cameras
    cv2.imshow('Regular Camera', regular_frame)
    cv2.imshow('Thermal Camera', thermal_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close windows
regular_cam.release()
thermal_cam.release()
cv2.destroyAllWindows()
