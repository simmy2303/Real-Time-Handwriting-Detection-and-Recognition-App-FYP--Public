import cv2
import numpy as np
import traceback
import logging
from ultralytics import YOLO
from inferenceModel import ImageToWordModel
# detect.py
# pip install "uvicorn[standard]", pip install fastapi, pip install -r requirements.txt
 
#autocorrect, pip install python-multipart, uvicorn app:app --host 0.0.0.0 --port 800
#cd "C:\Users\User\OneDrive - Sunway Education Group\Desktop\BSC (Hons) Computer Science\BSC sem 9\Capstone 2\sentence recognition\tensorflow\example"import cv2
from ultralytics import YOLO
from inferenceModel import ImageToWordModel
 
yolo_model = YOLO("models\my_model.pt")  # adjust to your .pt path
crnn_model = ImageToWordModel("models/model0.0488.onnx", "val.csv")
 
def detect_and_recognize_text(image):
    results = yolo_model(image, verbose=False)
    detections = results[0].boxes
    output = []
 
    for box in detections:
        conf = box.conf.item()
        if conf < 0.5:
            continue
 
        xmin, ymin, xmax, ymax = map(int, box.xyxy.squeeze().cpu().numpy())
        cropped = image[ymin:ymax, xmin:xmax]
 
        try:
            raw_text = crnn_model.predict(cropped)
            corrected_text = crnn_model.correct_spelling(raw_text)
        except Exception as e:
            corrected_text = "[error]"
            print(f"CRNN error: {e}")
 
        # Draw box and text on image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{corrected_text} ({round(conf, 2)})"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
 
        output.append({
            "bbox": [xmin, ymin, xmax, ymax],
            "text": corrected_text,
            "conf": round(conf, 2)
        })
 
    return output, image  # return modified image too