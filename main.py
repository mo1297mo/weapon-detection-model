from ultralytics import YOLO

model = YOLO("best.pt")  # Make sure best.pt is in the same folder

# Track with your MacBook's webcam
model.track(
    source=0,                # Webcam (built-in)
    tracker="botsort.yaml", # Built-in tracker
    show=True,              # Show live output
    save=True,              # Save to runs/track/predict/
    conf=0.3                # Confidence threshold
)
