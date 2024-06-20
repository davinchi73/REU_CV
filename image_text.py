import cv2
import pytesseract
import re

# Ensure Tesseract's executable is in your PATH or provide the path directly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open video stream for the thermal camera (assuming thermal camera is at index 1)
thermal_cam = cv2.VideoCapture(0)  # Adjust the index as per your setup

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated

def extract_text_from_image(image):
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except Exception as e:
        print(f"Error: An exception occurred during OCR: {e}")
        return ""

def extract_temperature_values(text):
    max_pattern = r'\bMax\b\D*(\d+(\.\d+)?)'
    min_pattern = r'\bMin\b\D*(\d+(\.\d+)?)'
    max_results = re.findall(max_pattern, text, flags=re.IGNORECASE)
    min_results = re.findall(min_pattern, text, flags=re.IGNORECASE)
    return max_results, min_results

while True:
    ret, frame = thermal_cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    processed_frame = preprocess_image(frame)
    text = extract_text_from_image(processed_frame)
    print("Extracted Text:\n", text)

    max_results, min_results = extract_temperature_values(text)
    
    if max_results:
        for result in max_results:
            max_value = result[0]
            print(f"Max: {max_value}")
    else:
        print("No Max results found.")

    if min_results:
        for result in min_results:
            min_value = result[0]
            print(f"Min: {min_value}")
    else:
        print("No Min results found.")

    # Display the original and processed frames
    cv2.imshow('Thermal Camera', frame)
    cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
thermal_cam.release()
cv2.destroyAllWindows()