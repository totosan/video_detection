from ultralytics import YOLO

# Load the YOLO12 model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx")
