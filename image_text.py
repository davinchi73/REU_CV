import cv2
import pytesseract
import re

# Ensure Tesseract's executable is in your PATH or provide the path directly
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
# Load the image
image_path = "thermal for meggie.png"
image = cv2.imread(image_path)

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

    # Print the extracted text
    print("Extracted Text:\n", text)

    # Define patterns for "max" and "min" with associated numbers
    max_pattern = r'\bMax\b\D*(\d+(\.\d+)?)'
    min_pattern = r'\bMin\b\D*(\d+(\.\d+)?)'

    # Find all occurrences of patterns in the text
    max_results = re.findall(max_pattern, text, flags=re.IGNORECASE)
    min_results = re.findall(min_pattern, text, flags=re.IGNORECASE)

    # Print results
    if max_results:
        print("Max results found:")
        for result in max_results:
            max_value = result[0]
            print(f"Max: {max_value}")
    else:
        print("No Max results found.")

    if min_results:
        print("Min results found:")
        for result in min_results:
            min_value = result[0]
            print(f"Min: {min_value}")
    else:
        print("No Min results found.")

except Exception as e:
    print(f"Error: An exception occurred during OCR: {e}")

# Display the original and processed images (optional)
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()