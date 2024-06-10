# import os

import cv2

import time

import sys

import numpy as np

# import random


# def randomcode1():
#     image = cv2.imread("SampleImages/antiqueTractors.jpg")
#     (bc, gc, rc) = cv2.split(image)

#     # each channel is shown as grayscale, because it only has value per pixel
#     cv2.imshow("Blue channel", bc)
#     cv2.imshow("Green channel", gc)
#     cv2.imshow("Red channel", rc)
#     cv2.moveWindow("Blue channel", 30, 30)
#     cv2.moveWindow("Green channel", 330, 60)
#     cv2.moveWindow("Red channel", 630, 90)
#     cv2.waitKey(0)

#     # Put image back together again
#     imCopy = cv2.merge((bc, gc, rc))
#     cv2.imshow("Image Copy", imCopy)
#     cv2.waitKey(0)


# def numpyRunner():
#     origImage = cv2.imread("SampleImages/snowLeo4.jpg")
#     gray = cv2.cvtColor(origImage, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Gray image", gray)

#     blankImg1 = np.zeros((400, 250), np.uint8)
#     cv2.imshow("Black background image", blankImg1)

#     blankImg2 = 255 * np.ones((300, 300), np.uint8)
#     cv2.imshow("White background image", blankImg2)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def slideshow():
#     for file in os.listdir("SampleImages"):
#         if file.lower().endswith(('.jpg', '.png', '.jpeg')):
#             path = os.path.join("SampleImages", file)
#             img = cv2.imread(path)

#             # Display the image
#             cv2.imshow(file, img)

#             # Wait for a key press
#             cv2.waitKey(0)

#         # Close all OpenCV windows
#         cv2.destroyAllWindows()


# def blendr():
#     mushroom = cv2.imread("SampleImages/mushrooms.jpg")
#     chicago = cv2.imread("SampleImages/chicago.jpg")

#     minWidth = min(mushroom.shape[0], chicago.shape[0])
#     minHeight = min(mushroom.shape[1], chicago.shape[1])

#     mushroomCrop = mushroom[:minWidth, :minHeight]
#     chicagoCrop = chicago[:minWidth, :minHeight]

#     wgt = 1.0

#     for x in range(0, 50):

#         blended = cv2.addWeighted(mushroomCrop, wgt, chicagoCrop, (1-wgt), 0)
#         cv2.imshow("", blended)
#         # time.sleep(0.01)
#         cv2.waitKey(0)
#         wgt -= 0.08

#     cv2.destroyAllWindows()


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



# def translation():
#     img = cv2.imread("SampleImages/snowLeo2.jpg")
#     (rows, cols, dep) = img.shape
#     cv2.imshow("Original", img)
#     transMatrix = np.float32([[1, 0, 30], [0, 1, 50]])  # change 30 and 50
#     transImag = cv2.warpAffine(img, transMatrix, (cols, rows))
#     cv2.imshow("Translated", transImag)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def rotation():
#     img = cv2.imread("SampleImages/snowLeo2.jpg")
#     cv2.imshow("Original", img)
#     (rows, cols, depth) = img.shape
#     rotMat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
#     rotImg = cv2.warpAffine(img, rotMat, (cols, rows))
#     cv2.imshow("Rotated", rotImg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def jitter():
#     img = cv2.imread("SampleImages/snowLeo2.jpg")
#     (rows, cols, dep) = img.shape
#     cv2.imshow("Original", img)

#     while True:
#         # Generate random offsets in x and y directions
#         random_offset_x = random.randint(-100, 100)
#         random_offset_y = random.randint(-100, 100)

#         transMatrix = np.float32([[1, 0, random_offset_x], [0, 1, random_offset_y]])  # Random offsets
#         transImag = cv2.warpAffine(img, transMatrix, (cols, rows))
#         cv2.imshow("Translated", transImag)

#         key = cv2.waitKey(50)  # Adjust the wait time as needed
#         if key != -1:  # Any key pressed breaks the loop
#             break

#     cv2.destroyAllWindows()


if __name__ == '__main__':
    # videoCap()
    # liveCap()
    # movementDetect()
    facePics()