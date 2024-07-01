import cv2


def crop_above(frame, zoom_factor, offset_y_pixels, offset_x_pixels):
    height, width = frame.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    
    # Adjust the starting row to account for the vertical offset due to camera position
    start_row = int((height - new_height) / 2) - offset_y_pixels
    start_row = max(0, start_row)  # Ensure the start row is not negative
    end_row = start_row + new_height
    if end_row > height:
        end_row = height
        start_row = height - new_height

    # Adjust the starting column to account for the horizontal offset
    start_col = int((width - new_width) / 2) - offset_x_pixels
    start_col = max(0, start_col)  # Ensure the start column is not negative
    end_col = start_col + new_width
    if end_col > width:
        end_col = width
        start_col = width - new_width

    return frame[start_row:end_row, start_col:end_col]


# Initialize cameras
regular_cam = cv2.VideoCapture(0)
thermal_cam = cv2.VideoCapture(1)


zoom_factor = 2.5  # Factor by which we are zooming in
offset_y_inches = -0.8
offset_x_inches = 1  # Adjust this value as needed for a slight horizontal offset
dpi = 30  # Assuming 30 pixels per inch (adjust according to your screen/camera setup)
offset_y_pixels = int(offset_y_inches * dpi)
offset_x_pixels = int(offset_x_inches * dpi)


while True:
    ret1, regular_frame = regular_cam.read()
    ret2, thermal_frame = thermal_cam.read()

    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    # Crop the regular camera frame with vertical and horizontal offsets
    zoomed_frame = crop_above(regular_frame, zoom_factor, offset_y_pixels, offset_x_pixels)
    
    # Resize the cropped frame to match the thermal frame size
    thermal_height, thermal_width = thermal_frame.shape[:2]
    zoomed_frame_resized = cv2.resize(zoomed_frame, (thermal_width, thermal_height))

    # Display the frames
    cv2.imshow('Regular Camera', zoomed_frame_resized)
    cv2.imshow('Thermal Camera', thermal_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture objects and close windows
regular_cam.release()
thermal_cam.release()
cv2.destroyAllWindows()