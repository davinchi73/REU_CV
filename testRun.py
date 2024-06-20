import os

import cv2

import time

import sys

import numpy as np

#------------

def videoCap():

    # cascPath = sys.argv[1]
    noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    # vidCap = cv2.VideoCapture(0)
    regVid = cv2.VideoCapture("testVideoREU.mp4")
    tVid = cv2.VideoCapture("testVideoREU.mp4")

    while True:
        ret, img = regVid.read()
        t_ret, t_img = tVid.read()

        if ret == False:
            break
        elif t_ret == False:
            break
        
        # resize for optimization
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        t_img = cv2.resize(t_img, (0, 0), fx=0.5, fy=0.5)

        t_img = apply_thermal_effect(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nose = noseCascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the nose
        for (x, y, w, h) in nose:
            cv2.rectangle(t_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(t_img, "Nose", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("t_vid", t_img)
        cv2.imshow("vid", img)

        x = cv2.waitKey(10) # if 0 will be a still frame
        ch = chr(x & 0xFF)  # bitwise and that removes all bits above 255

        if ch == "q":
            break
        

    # break and destroy
    cv2.destroyAllWindows()
    regVid.release()
    tVid.release()


def liveCap():

    # cascPath = sys.argv[1]
    noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    # vidCap = cv2.VideoCapture(0)
    regVid = cv2.VideoCapture(0)

    while True:
        ret, img = regVid.read()
        t_ret, t_img = regVid.read()

        if ret == False:
            break
        elif t_ret == False:
            break
        

        t_img = apply_thermal_effect(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nose = noseCascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the nose
        for (x, y, w, h) in nose:
            cv2.rectangle(t_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(t_img, "Nose", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("t_vid", t_img)
        cv2.imshow("vid", img)

        x = cv2.waitKey(10) # if 0 will be a still frame
        ch = chr(x & 0xFF)  # bitwise and that removes all bits above 255

        if ch == "q":
            break

    # break and destroy
    cv2.destroyAllWindows()
    regVid.release()


def apply_thermal_effect(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply colormap for thermal effect
    thermal_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET) # just a filter
    
    return thermal_img
    

def movementDetect():
    noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    regVid = cv2.VideoCapture(0)

    noseCoords = []
    zoomed_in = False
    zoom_start_time = None
    zoom_factor = 3 # bigger num = less zoom
    movement_threshold = 30 # bigger num = more movement required

    while True:
        # read vid footage
        ret, img = regVid.read()

        # check if ret false
        if ret == False:
            break
            
        # gray scale and nose detection setup
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nose = noseCascade.detectMultiScale(gray, 1.3, 5)

        # if first nose detection
        if len(nose) > 0:
            x, y, w, h = nose[0]  # Consider only the first detected nose
            noseCoords.append((x, y))

            if len(noseCoords) > 30:  # Check for stability over 2 seconds (assuming 30 fps)
                if all(abs(x - nx) < 50 and abs(y - ny) < 50 for nx, ny in noseCoords[-50:]):
                    if not zoomed_in:
                        zoom_start_time = time.time()
                    zoomed_in = True
                else:
                    zoomed_in = False

        if zoomed_in:
            zoomed_w = int(w * zoom_factor)
            zoomed_h = int(h * zoom_factor)
            zoomed_x = max(0, x + (w - zoomed_w) // 2)
            zoomed_y = max(0, y + (h - zoomed_h) // 2)
            zoomed_img = img[zoomed_y:zoomed_y+zoomed_h, zoomed_x:zoomed_x+zoomed_w]
            img = cv2.resize(zoomed_img, (img.shape[1], img.shape[0]))  # Resize to original size

            # Check for movement
            if detect_nose_movement(noseCoords) > 50:
                zoomed_in = False
                noseCoords = []  # Reset nose coordinates

        if not zoomed_in:
            # Draw rectangle around nose
            for (x, y, w, h) in nose:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Nose", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("vid", img)

        x = cv2.waitKey(10)
        ch = chr(x & 0xFF)

        if ch == "q":
            break

    cv2.destroyAllWindows()
    regVid.release()


def detect_nose_movement(noseCoords):
    # Check for movement by comparing current nose position with previous positions
    movement = sum((abs(x - prev_x) > 50 or abs(y - prev_y) > 50) for (x, y) in noseCoords for (prev_x, prev_y) in noseCoords[:-1])
    return movement


def facePics():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    regVid = cv2.VideoCapture(0)
    
    while True:

        ret, img = regVid.read()

        # check if ret false
        if ret == False:
            break
            
        # gray scale and nose detection setup
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("vid", img)

        x = cv2.waitKey(10) # if 0 will be a still frame
        ch = chr(x & 0xFF)  # bitwise and that removes all bits above 255

        if ch == "q":
            break
        elif ch == "p":
            for (x, y, w, h) in face:
                face_img = img[y:y+h, x:x+w]
                cv2.imshow(f"Face at ({x},{y})", face_img)

    # break and destroy
    cv2.destroyAllWindows()
    regVid.release()


def faceBlink():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    regVid = cv2.VideoCapture(0)

    isBlinking = False
    isRed = False
    isGreen = True
    frameCount = 0
    
    while True:

        ret, img = regVid.read()

        # check if ret false
        if not ret:
            break
            
        # gray scale and nose detection setup
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.3, 5)


        for (x, y, w, h) in face:
            if not isBlinking:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            elif isBlinking:
                if frameCount > 5:
                    if isRed:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        isRed = False
                        isGreen = True
                        frameCount = 0 
                    else:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        isRed = True
                        isGreen = False
                        frameCount = 0
                else:
                    if isRed:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    elif isGreen:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        cv2.imshow("vid", img)

        x = cv2.waitKey(10) # if 0 will be a still frame
        ch = chr(x & 0xFF)  # bitwise and that removes all bits above 255

        if ch == "q":
            break
        elif ch == "p":
            if not isBlinking:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                isRed = True
                isGreen = False
                isBlinking = True
            elif isBlinking:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                isBlinking = False
                isRed = False
                isGreen = True
                isBlinking = False

        if isBlinking:
            frameCount += 1

    # break and destroy
    cv2.destroyAllWindows()
    regVid.release()



def liveTestCap():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    thermalVid = cv2.VideoCapture(0)  # Assuming thermal camera is at index 1

    frame_count = 0
    avg_bgr = None
    buffer_size = 10
    color_buffer = []

    while True:
        ret, img = thermalVid.read()

        if not ret:
            break

        # Convert image to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Approximate the area for breathing (mouth/nose region)
            # Example: You can draw a rectangle slightly below the midpoint of the face box
            # Adjust these coordinates based on where you think the mouth/nose region is
            breathing_area_top_left = (x + int(w * 0.35), y + int(h * 0.6) - 40)
            breathing_area_bottom_right = (x + int(w * 0.65), (y + h) - 40)
            cv2.rectangle(img, breathing_area_top_left, breathing_area_bottom_right, (0, 255, 0), 2)
            cv2.putText(img, "Breathing Area", (x + int(w * 0.25), y + int(h * 0.6) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
            # Extract pixels from breathing area box
            breathing_area = img[breathing_area_top_left[1]:breathing_area_bottom_right[1],
                                 breathing_area_top_left[0]:breathing_area_bottom_right[0]]

            # Calculate average BGR color every 10 frames
            if frame_count % buffer_size == 0:
                color_buffer.append(np.mean(breathing_area, axis=(0, 1)))
                if len(color_buffer) > buffer_size:
                    color_buffer.pop(0)
                avg_bgr = np.mean(color_buffer, axis=0).astype(int)

            # Display average BGR color on the image
            if avg_bgr is not None:
                cv2.putText(img, f"Avg BGR: {avg_bgr}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Display the result
        cv2.imshow("Thermal Camera with Detection", img)

        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Increment frame count
        frame_count += 1

    # Clean up
    cv2.destroyAllWindows()
    thermalVid.release()


#-----

if __name__ == '__main__':
    # videoCap()
    # liveCap()
    # movementDetect()
    # facePics()
    # faceBlink()
    liveTestCap()