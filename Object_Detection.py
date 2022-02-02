import cv2
from deepstack_sdk import ServerConfig, Detection


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


if __name__ == "__main__":
    config = ServerConfig("http://localhost:80")
    detection = Detection(config)
    capture = cv2.VideoCapture(0)

    while (True):
        ret, frame = capture.read()
        if ret:
            detections = detection.detectObject(frame, output=None)
            frame = draw_detections(frame, detections)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
