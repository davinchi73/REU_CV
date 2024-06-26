import cv2
import numpy as np
from flask import Flask, render_template, Response, request

app = Flask(__name__)

isBlinking = False  # Global variable to hold the blinking state

@app.route('/')
def index():
    return render_template('index1.html')

def generate_face_grid():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_images = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_images.append(face_img)

        num_faces = len(face_images)
        rows = (num_faces + 1) // 2
        combined_height = max(480, rows * 240)
        combined_faces = np.zeros((combined_height, 960, 3), np.uint8)

        frame_resized = cv2.resize(frame, (640, 480))
        combined_faces[:480, :640] = frame_resized

        for idx, face_img in enumerate(face_images):
            face_resized = cv2.resize(face_img, (320, 240))
            row = idx // 2
            col = idx % 2
            y_start = row * 240
            y_end = (row + 1) * 240
            x_start = 640 + col * 320
            x_end = 640 + (col + 1) * 320

            if x_end <= combined_faces.shape[1] and y_end <= combined_faces.shape[0]:
                combined_faces[y_start:y_end, x_start:x_end] = face_resized

        ret, jpeg = cv2.imencode('.jpg', combined_faces)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def generate_face_blink():
    global isBlinking
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    regVid = cv2.VideoCapture(0)

    isRed = False
    frameCount = 0

    while True:
        ret, img = regVid.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if not isBlinking:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                if frameCount > 5:
                    isRed = not isRed
                    frameCount = 0
                color = (0, 0, 255) if isRed else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, "Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        frameCount += 1

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    regVid.release()

@app.route('/video_feed')
def video_feed():
    func = request.args.get('function', default='grid', type=str)
    toggle = request.args.get('toggleBlinking', default='false', type=str)
    if toggle == 'true':
        global isBlinking
        isBlinking = not isBlinking

    if func == 'blink':
        return Response(generate_face_blink(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_face_grid(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)