import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Extract faces and prepare the combined image
        face_images = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_images.append(face_img)

        # Calculate the required height for the combined_faces image
        num_faces = len(face_images)
        rows = (num_faces + 1) // 2  # Number of rows needed
        combined_height = max(480, rows * 240)
        combined_faces = np.zeros((combined_height, 960, 3), np.uint8)

        # Place the main frame on the left side
        frame_resized = cv2.resize(frame, (640, 480))
        combined_faces[:480, :640] = frame_resized

        # Place the face grid on the right side
        for idx, face_img in enumerate(face_images):
            face_resized = cv2.resize(face_img, (320, 240))
            row = idx // 2
            col = idx % 2
            y_start = row * 240
            y_end = (row + 1) * 240
            x_start = 640 + col * 320
            x_end = 640 + (col + 1) * 320

            # Debugging statements
            print(f"Placing face {idx + 1}/{num_faces}:")
            print(f"  Face resized shape: {face_resized.shape}")
            print(f"  Target slice: [{y_start}:{y_end}, {x_start}:{x_end}]")
            print(f"  Combined_faces shape: {combined_faces.shape}")

            # Check if the target slice is within bounds
            if x_end <= combined_faces.shape[1] and y_end <= combined_faces.shape[0]:
                combined_faces[y_start:y_end, x_start:x_end] = face_resized
            else:
                print(f"  Skipping face {idx + 1} due to slice bounds.")

        # Encode the image as a JPEG and return it as a byte stream
        ret, jpeg = cv2.imencode('.jpg', combined_faces)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)