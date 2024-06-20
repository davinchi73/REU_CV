import cv2
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Nose detection xml
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Thermal camera video capture
thermal_cam = cv2.VideoCapture(0)  # Thermal camera

frame_counter = 0

def min_and_max(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enhance contrast using histogram equalization
    gray = cv2.equalizeHist(gray)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
        print(f"Max Results: {max_results}, Min Results: {min_results}")  # Debugging statement to check pattern matches

        # Extract max and min values
        max_value = float(max_results[0][0]) if max_results else None
        min_value = float(min_results[0][0]) if min_results else None
        return max_value, min_value 
    
    except Exception as e:
        print(f"Error: An exception occurred during OCR: {e}")
        return None, None
    
# Function to convert pixel value to temperature (temperature is in Celsius) 
def pixel_to_temperature(pixel_value, min_temp=20, max_temp=100):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)

while True:
    frame_counter += 1
    # Capture frame-by-frame from the thermal camera
    ret, thermal_frame = thermal_cam.read()

    # Make sure that the frame is read properly
    if not ret:
        print("Failed to grab frames")
        break
    
    # Only extract the temperature every 10 frames 
    if frame_counter % 10 == 0: 
        max_temp, min_temp = min_and_max(thermal_frame)
        if max_temp is None or min_temp is None:
            print("Failed to extract temperature range")
            continue

        # Convert the thermal frame to grayscale and detect faces (assuming the entire frame is the face)
        gray_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        noses = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # For coordinates in the noses detected
        for (nx, ny, nw, nh) in noses:
            nose_center_x = nx + nw // 2
            nose_center_y = ny + nh // 2
            cv2.rectangle(thermal_frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
            cv2.circle(thermal_frame, (nose_center_x, nose_center_y), 5, (0, 0, 255), -1)

            # Ensure thermal_frame is in single-channel (grayscale)
            if len(thermal_frame.shape) == 3 and thermal_frame.shape[2] == 3:
                thermal_frame_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
            else:
                thermal_frame_gray = thermal_frame

            # Assuming that the nose is detected correctly
            nose_temperature_pixel_value = thermal_frame_gray[nose_center_y, nose_center_x]
            nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=min_temp, max_temp=max_temp)
            
            # Say the temperature at the nose if the max and min temperatures are not none 
            if max_temp is not None and min_temp is not None:
                nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=min_temp, max_temp=max_temp)
                print('Temperature at nose:', nose_temperature)
                cv2.putText(thermal_frame, f'{nose_temperature:.2f} °C', (nose_center_x, nose_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the thermal camera frame
    cv2.imshow('Thermal Camera', thermal_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
thermal_cam.release()
cv2.destroyAllWindows()