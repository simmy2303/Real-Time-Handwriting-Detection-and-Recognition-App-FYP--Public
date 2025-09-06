# yolo_detect.py
import cv2
import numpy as np
from ultralytics import YOLO
from inferenceModel import ImageToWordModel

def pipeline_generator(
    source=0,
    thresh=0.5,
    resolution=(640, 480),
    buffer_size=8
):
    # load models
    yolo = YOLO("models/my_model.pt")
    crnn = ImageToWordModel("models/model0.0488.onnx", {"vocab": "..."})

    # open camera
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # rolling FPS buffer (optional)
    frame_rate_buffer = []

    while True:
        start = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            break

        # 1) YOLO detection
        results = yolo.predict(frame, conf=thresh)[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        # 2) CRNN on each crop
        annotated = frame.copy()
        for x1, y1, x2, y2 in boxes:
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            text = crnn.predict(crop)
            cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(annotated, text, (int(x1),int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 3) encode to JPEG
        ok, jpeg = cv2.imencode('.jpg', annotated)
        if not ok:
            continue

        # yield multipart chunk
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            jpeg.tobytes() +
            b'\r\n'
        )

        # optional FPS calc
        end = cv2.getTickCount()
        time_spent = (end - start)/cv2.getTickFrequency()
        frame_rate_buffer.append(1.0/time_spent)
        if len(frame_rate_buffer) > buffer_size:
            frame_rate_buffer.pop(0)

    cap.release()
