import cv2
import numpy as np
import pytesseract
import re


def liveTestCapThermo():

    # initialize pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # load face cascade and thermal camera live footage
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    thermalVid = cv2.VideoCapture(0) # adjust value if needed

    # create variables for temp detection later
    frame_count = 0
    avg_bgr = None
    buffer_size = 90
    color_buffer = []

    # while true, read thermal cam footage
    while True:
        frame_count += 1
        ret, img = thermalVid.read()

        if not ret:
            print("Failed to grab frames")
            break

        if frame_count % buffer_size == 0:
            max_temp, min_temp = min_and_max(img)
            if max_temp is None or min_temp is None:
                print("Failed to extract temperature range")
                continue

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            breathing_area_top_left = (x + int(w * 0.35), y + int(h * 0.6) - 40)
            breathing_area_bottom_right = (x + int(w * 0.65), (y + h) - 40)
            cv2.rectangle(img, breathing_area_top_left, breathing_area_bottom_right, (0, 255, 0), 2)
            cv2.putText(img, "Breathing Area", (x + int(w * 0.25), y + int(h * 0.6) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            breathing_area = img[breathing_area_top_left[1]:breathing_area_bottom_right[1],
                                        breathing_area_top_left[0]:breathing_area_bottom_right[0]]

            if frame_count % buffer_size == 0:
                color_buffer.append(np.mean(breathing_area, axis=(0, 1)))
                if len(color_buffer) > buffer_size:
                    color_buffer.pop(0)
                avg_bgr = np.mean(color_buffer, axis=0).astype(int)

                if avg_bgr is not None:
                    avg_pixel_value = np.mean(avg_bgr)
                    nose_temperature = pixel_to_temperature(avg_pixel_value, min_temp=min_temp, max_temp=max_temp)
                    print('Temperature at nose:', nose_temperature)
                    cv2.putText(img, f'{nose_temperature:.2f} °C', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow('Thermal Camera with Detection', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # img.release()
    cv2.destroyAllWindows()


# Function to convert pixel value to temperature (temperature is in Celsius) 
def pixel_to_temperature(pixel_value, min_temp, max_temp):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)


def min_and_max(image):
    # Define the ROI for the area where the min and max values are displayed
    roi = image[0:150, 0:250]  # Adjust these values based on your specific camera feed

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)

        max_pattern = r'\bMax\b\D*(\d+(\.\d+)?)'
        min_pattern = r'\bMin\b\D*(\d+(\.\d+)?)'
        max_results = re.findall(max_pattern, text, flags=re.IGNORECASE)
        min_results = re.findall(min_pattern, text, flags=re.IGNORECASE)
        print(f"Max Results: {max_results}, Min Results: {min_results}")

        max_value = float(max_results[0][0]) if max_results else None
        min_value = float(min_results[0][0]) if min_results else None
        return max_value, min_value

    except Exception as e:
        print(f"Error: An exception occurred during OCR: {e}")
        return None, None


if __name__ == '__main__':
    liveTestCapThermo()