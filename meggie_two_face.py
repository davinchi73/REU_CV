import cv2
import numpy as np
import winsound  # Import the winsound module for playing alert sound
import time  # Import the time module for tracking detection times

# face and nose detection xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# video for both cameras
regular_cam = cv2.VideoCapture(1)  # Regular camera
thermal_cam = cv2.VideoCapture(0)  # Thermal camera

last_extraction_time = time.time()
flash_counter = 0  # Counter for flashing effect

class Person:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox  # Bounding box (x, y, w, h)
        self.temperatures = [] #Keep track of all the previous last temperatures
        self.last_detected = time.time()  # Initialize last detected time
        self.alarm_on = False
        self.alarm_start_time = None  #initialize start time 
        self.avg_temps = []  # Store average temperatures over 20 seconds
        self.previous_avg_temp = None

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.last_detected = time.time()  # Update last detected time

    def add_temperature(self, temp):
        self.temperatures.append(temp)
        if len(self.temperatures) > 10:
            self.temperatures.pop(0)  # Keep only the last 20 temperatures

        # Calculate average temperature over the last 20 seconds
        if len(self.temperatures) == 10:
            current_avg_temp = sum(self.temperatures) / 10
            self.avg_temps.append(current_avg_temp)
            if len(self.avg_temps) > 2:
                self.avg_temps.pop(0)  # Keep only the last 2 average temperatures

    def check_death(self):
        if len(self.avg_temps) == 2:
            if abs(self.avg_temps[-1] - self.avg_temps[-2]) <= 0.2:
                return True
        return False

    def get_last_temperature(self):
        return self.temperatures[-1] if self.temperatures else None

# List to store persons and increment ID 
persons = []
next_person_id = 0

def play_alert_sound():
    winsound.Beep(1000, 500)

# Function to convert pixel value to temperature (temperature is in Celsius)
def pixel_to_temperature(pixel_value, min_temp=20, max_temp=100):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)

def get_person_id(bbox):
    global next_person_id
    x, y, w, h = bbox
    for person in persons:
        px, py, pw, ph = person.bbox
        if abs(x - px) < 75 and abs(y - py) < 75:
            return person.id
    person = Person(next_person_id, bbox)
    persons.append(person)
    next_person_id += 1
    return person.id

# While there is an active video frame
while True:
    current_time = time.time() 
    # Capture frame-by-frame from both cameras
    ret1, regular_frame = regular_cam.read()
    ret2, thermal_frame = thermal_cam.read()

    # Make sure that ret1 and ret2 are reading properly
    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    # Only extract the temperature every 3 seconds
    if current_time - last_extraction_time >= 2:
        last_extraction_time = current_time  # Reset the extraction timer for the person instance

        # Convert the regular frame to grayscale and detect faces
        gray_frame = cv2.cvtColor(regular_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # For coordinates in the faces detected
        for (x, y, w, h) in faces:
            person_id = get_person_id((x, y, w, h))
            person = [p for p in persons if p.id == person_id][0]
            person.update_bbox((x, y, w, h))

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

                # Assuming that both cameras are on the same plane
                nose_temperature_pixel_value = thermal_frame_gray[nose_center_y, nose_center_x]
                nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=17.2, max_temp=33.4)

                # Get person ID and update their temperature
                person_id = get_person_id((x, y, w, h))
                person = next(person for person in persons if person.id == person_id)
                person.add_temperature(nose_temperature)
                print(f'Temperature at nose for person {person_id}: {nose_temperature:.2f} °C')
                cv2.putText(regular_frame, f'{nose_temperature:.2f} °C', (nose_center_x, nose_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Detecting if the average temperature over the last 20 seconds hasn't changed more than 0.2 Celsius for any person
        for person in persons:
            if person.check_death():
                print(f"Death detected for person {person.id}!")
                if not person.alarm_on:
                    play_alert_sound()  # Play alert sound
                    person.alarm_on = True
                    person.alarm_start_time = time.time()  # Add this line
            else:
                person.alarm_on = False

    # Draw flashing circles if alarm is on for person class
    for person in persons:
        flash_counter += 1
        if person.alarm_on:
            #if the alarm has been on for 10 seconds stop the flashing and set alarm to False
            if person.alarm_start_time and (time.time() - person.alarm_start_time > 10):
                person.alarm_on = False  # Turn off alarm after 10 seconds
                person.alarm_start_time = None
                continue
            x, y, w, h = person.bbox
            # Alternate circle color between red and green
            color = (0, 0, 255) if flash_counter % 20 < 10 else (0, 255, 0)
            cv2.circle(regular_frame, (x + w // 2, y + h // 2), max(w, h) // 2, color, 4)
    
    #delete person from the persons list if they haven't been detected in 10 seconds 
    persons = [person for person in persons if current_time - person.last_detected <= 10]

    # Show the frames of the cameras
    cv2.imshow('Regular Camera', regular_frame)
    cv2.imshow('Thermal Camera', thermal_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close windows
regular_cam.release()
thermal_cam.release()
cv2.destroyAllWindows()
