from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
model = YOLO('my_model.pt')  
model.info()               # Prints backbone/neck/head block counts
print(model.model)         # Dumps the full layer-by-layer table
