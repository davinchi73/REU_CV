from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import winsound
import time

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

regular_cam = cv2.VideoCapture(1)
thermal_cam = cv2.VideoCapture(0)

last_extraction_time = time.time()
flash_counter = 0

class Person:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox
        self.temperatures = []
        self.last_detected = time.time()
        self.alarm_on = False
        self.alarm_start_time = None
        self.previous_avg_temp = None
        self.death_detected_time = None

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.last_detected = time.time()

    def add_temperature(self, temp):
        self.temperatures.append(temp)
        if len(self.temperatures) > 20:
            self.temperatures.pop(0)

    def get_average_temperature(self):
        if len(self.temperatures) == 20:
            return sum(self.temperatures) / 20
        return None

    def check_death(self):
        current_avg_temp = self.get_average_temperature()
        if current_avg_temp is not None and self.previous_avg_temp is not None:
            if abs(current_avg_temp - self.previous_avg_temp) <= 0.2:
                if self.death_detected_time is None:
                    self.death_detected_time = time.time()
                return True
        self.previous_avg_temp = current_avg_temp
        self.death_detected_time = None
        return False

persons = []
next_person_id = 0

def pixel_to_temperature(pixel_value, min_temp=20, max_temp=100):
    return min_temp + (pixel_value / 255) * (max_temp - min_temp)

def get_person_id(bbox):
    global next_person_id
    x, y, w, h = bbox
    for person in persons:
        px, py, pw, ph = person.bbox
        if abs(x - px) < 75 and abs(y - py) < 75:
            return person.id
    return None

def generate_frames():
    global last_extraction_time, flash_counter, persons, next_person_id
    while True:
        current_time = time.time()
        ret1, regular_frame = regular_cam.read()
        ret2, thermal_frame = thermal_cam.read()

        if not ret1 or not ret2:
            print("could not grab frames")
            break

        if current_time - last_extraction_time >= 3:
            last_extraction_time = current_time

            rgb_frame = cv2.cvtColor(regular_frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = rgb_frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    person_id = get_person_id((x, y, w, h))
                    if person_id is None:
                        person_id = next_person_id
                        persons.append(Person(person_id, (x, y, w, h)))
                        next_person_id += 1
                    person = [p for p in persons if p.id == person_id][0]
                    person.update_bbox((x, y, w, h))

                    face_results = face_mesh.process(rgb_frame)
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            nose_tip = face_landmarks.landmark[1]
                            nose_center_x = int(nose_tip.x * iw)
                            nose_center_y = int(nose_tip.y * ih)

                            cv2.rectangle(regular_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.circle(regular_frame, (nose_center_x, nose_center_y), 5, (0, 0, 255), -1)

                            if len(thermal_frame.shape) == 3 and thermal_frame.shape[2] == 3:
                                thermal_frame_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
                            else:
                                thermal_frame_gray = thermal_frame

                            nose_temperature_pixel_value = thermal_frame_gray[nose_center_y, nose_center_x]
                            nose_temperature = pixel_to_temperature(nose_temperature_pixel_value, min_temp=17.2, max_temp=33.4)

                            person.add_temperature(nose_temperature)
                            print(f'Temperature at nose for person {person_id}: {nose_temperature:.2f} °C')
                            cv2.putText(regular_frame, f'{nose_temperature:.2f} °C', (nose_center_x, nose_center_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for person in persons:
                if person.check_death():
                    print(f"Death detected for person {person.id}!")
                    if not person.alarm_on:
                        winsound.Beep(1000, 500)
                        person.alarm_on = True
                        person.alarm_start_time = time.time()
                else:
                    person.alarm_on = False

        flash_counter += 1
        for person in persons:
            if person.death_detected_time and (time.time() - person.death_detected_time <= 30):
                x, y, w, h = person.bbox
                color = (0, 0, 255) if flash_counter % 20 < 10 else (0, 255, 0)
                cv2.circle(regular_frame, (x + w // 2, y + h // 2), max(w, h) // 2, color, 4)

        persons = [person for person in persons if current_time - person.last_detected <= 10]

        ret, buffer = cv2.imencode('.jpg', regular_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)