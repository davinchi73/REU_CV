import cv2
import numpy as np
import pytesseract
import re
import winsound  # Import the winsound module for playing alert sound


# Initialize pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load face cascade and thermal camera live footage
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
thermalVid = cv2.VideoCapture(0)  # Adjust value if needed

def play_alert_sound():
    winsound.Beep(1000, 500)  # Beep sound plays 


def liveTestCapThermo():
    frame_count = 0
    last_temp = 0
    avg_bgr = None
    buffer_size = 90 #check every 3 seconds 
    color_buffer = []
    last_face_box = None
    alarm_on = False

    while True:
        ret, img = thermalVid.read()

        if not ret:
            print("Failed to grab frames")
            break

        max_temp, min_temp = min_and_max(img)
        if max_temp is None or min_temp is None:
            print("Failed to extract temperature range")
            continue

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Reset last_face_box when new faces are detected
            last_face_box = None

        for (x, y, w, h) in faces:
            # Update last_face_box with current detection
            last_face_box = (x, y, w, h)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Define breathing area ROI within the detected face region
            breathing_area_top_left = (x + int(w * 0.35), y + int(h * 0.6))
            breathing_area_bottom_right = (x + int(w * 0.65), y + h)
            cv2.rectangle(img, breathing_area_top_left, breathing_area_bottom_right, (0, 255, 0), 2)
            cv2.putText(img, "Breathing Area", (x + int(w * 0.25), y + int(h * 0.6) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Extract breathing area from the thermal image
            breathing_area = img[breathing_area_top_left[1]:breathing_area_bottom_right[1],
                                        breathing_area_top_left[0]:breathing_area_bottom_right[0]]

            color_buffer.append(np.mean(breathing_area, axis=(0, 1)))
            if len(color_buffer) > buffer_size:
                color_buffer.pop(0)
            avg_bgr = np.mean(color_buffer, axis=0).astype(int)

            if avg_bgr is not None:
                avg_pixel_value = np.mean(avg_bgr)
                nose_temperature = pixel_to_temperature(avg_pixel_value, min_temp=min_temp, max_temp=max_temp)
                print('Temperature at nose:', nose_temperature)
                cv2.putText(img, f'{nose_temperature:.2f} °C', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # If no faces detected, use last known face box if available to draw
        if last_face_box is not None:
            (x, y, w, h) = last_face_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face (Last Known)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Define breathing area ROI within the last known face region
            breathing_area_top_left = (x + int(w * 0.35), y + int(h * 0.6))
            breathing_area_bottom_right = (x + int(w * 0.65), y + h)
            cv2.rectangle(img, breathing_area_top_left, breathing_area_bottom_right, (0, 255, 0), 2)
            cv2.putText(img, "Breathing Area (Last Known)", (x + int(w * 0.25), y + int(h * 0.6) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if abs(last_temp - nose_temperature) < 0.4: 
            print("Death detected!")
            if not alarm_on:
                play_alert_sound()  # Play alert sound
                alarm_on = True
            else:
                alarm_on = False

        last_temp = nose_temperature #set last_temp to be the recently detected nose temperature 
        #show breathing area on screen 
        cv2.imshow('Thermal Camera with Detection', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    thermalVid.release()
    cv2.destroyAllWindows()

# Function to convert pixel value to temperature (temperature is in Celsius)
def pixel_to_temperature(pixel_value, min_temp, max_temp):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)

# Function to extract min and max temperatures from image
def min_and_max(image):
    # Define the ROI for the area where the min and max values are displayed
    roi = image[0:150, 0:250] 

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)

        max_pattern = r'\bMax\b\D*(\d+(\.\d+)?)'
        min_pattern = r'\bMin\b\D*(\d+(\.\d+)?)'
        max_results = re.findall(max_pattern, text, flags=re.IGNORECASE)
        min_results = re.findall(min_pattern, text, flags=re.IGNORECASE)
        #print(f"Max Results: {max_results}, Min Results: {min_results}")

        max_value = float(max_results[0][0]) if max_results else None
        min_value = float(min_results[0][0]) if min_results else None
        return max_value, min_value

    except Exception as e:
        print(f"Error: An exception occurred during OCR: {e}")
        return None, None

if __name__ == '__main__':
    liveTestCapThermo()