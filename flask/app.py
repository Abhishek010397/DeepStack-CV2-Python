from flask import Flask, render_template, Response
import cv2
from deepstack_sdk import ServerConfig, Detection

config = ServerConfig("http://localhost:80")
detection = Detection(config)
app = Flask(__name__)
camera = cv2.VideoCapture(0)

def draw_detections(img, detections):
    for detection in detections:
        output_font_scale = 0.8e-3 * img.shape[0]
        label = detection.label
        img = cv2.rectangle(
            img,
            (detection.x_min, detection.y_min),
            (detection.x_max, detection.y_max),
            (0, 146, 224),
            2
        )
        img = cv2.putText(
            img=img,
            text=label + " ( " + str(100 * detection.confidence) + "% )",
            org=(detection.x_min - 10, detection.y_min - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=output_font_scale,
            color=(0, 146, 224),
            thickness=2
        )
    return img



def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            detections = detection.detectObject(frame, output=None)
            frame = draw_detections(frame, detections)
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
