import cv2

def find_camera_indices():
    index = 0
    available_indices = []
    while index < 10:  # You can increase this number if you have more cameras
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            available_indices.append(index)
            cap.release()
        index += 1
    return available_indices

print("Available camera indices: ", find_camera_indices())