from ultralytics import YOLO

modelname = "yolo12n"  # Model name

# Load a YOLO11n PyTorch model
model = YOLO(f"{modelname}.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO(f"{modelname}.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")