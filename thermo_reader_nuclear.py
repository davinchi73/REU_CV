import cv2
import numpy as np
import pytesseract
import re
import winsound  # Import the winsound module for playing alert sound

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# face and nose detection xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# video for both cameras
regular_cam = cv2.VideoCapture(1)  # Regular camera
thermal_cam = cv2.VideoCapture(0)  # Thermal camera

frame_counter = 0
last_temp = None
alarm_on = False  # Initial state of the alarm

# alert sound function when breathing temperature does not change
def play_alert_sound():
    winsound.Beep(1000, 500)

# Function to convert pixel value to temperature (temperature is in Celsius)
def pixel_to_temperature(pixel_value, min_temp=20, max_temp=100):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)

# while there is an active video frame
while True:
    frame_counter += 1
    last_nose_box = None
    nose_temperature = None
    # Capture frame-by-frame from both cameras
    ret1, regular_frame = regular_cam.read()
    ret2, thermal_frame = thermal_cam.read()

    # make sure that ret1 and ret2 are reading properly
    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    # only extract the temperature every 10 frames
    if frame_counter % 90 == 0:
        # Convert the regular frame to grayscale and detect faces
        gray_frame = cv2.cvtColor(regular_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # for coordinates in the faces detected
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            # Detect nose within the face ROI
            noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (nx, ny, nw, nh) in noses:
                last_nose_box = (nx, ny, nw, nh)
                nose_center_x = x + nx + nw // 2
                nose_center_y = y + ny + nh // 2
                cv2.rectangle(regular_frame, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (0, 255, 0), 2)
                cv2.circle(regular_frame, (nose_center_x, nose_center_y), 5, (0, 0, 255), -1)

                # Ensure thermal_frame is in single-channel (grayscale)
                if len(thermal_frame.shape) == 3 and thermal_frame.shape[2] == 3:
                    thermal_frame_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
                else:
                    thermal_frame_gray = thermal_frame

                # assuming that both cameras are on the same plane
                nose_temperature_pixel_value = thermal_frame_gray[nose_center_y, nose_center_x]
                nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=17.2, max_temp=33.4)

                # say the temperature at the nose if the max and min temperatures are not None
                nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=17.2, max_temp=33.4)
                print('Temperature at nose:', nose_temperature)
                cv2.putText(regular_frame, f'{nose_temperature:.2f} °C', (nose_center_x, nose_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # detecting if the change in temperature is less than 0.4 Celsius
        if nose_temperature is not None and last_temp is not None:
            if abs(last_temp - nose_temperature) < 0.4:
                print("Death detected!")
                if not alarm_on:
                    play_alert_sound()  # Play alert sound
                    alarm_on = True
            else:
                alarm_on = False

        last_temp = nose_temperature

    else:
        continue

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