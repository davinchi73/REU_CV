import cv2
import numpy as np

# face and nose detection xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# video for both cameras
regular_cam = cv2.VideoCapture(0)  # Regular camera
thermal_cam = cv2.VideoCapture(1)  # Thermal camera

# Function to convert pixel value to temperature
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
            nose_temperature = pixel_to_temperature(nose_temperature_pixel_value)
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
