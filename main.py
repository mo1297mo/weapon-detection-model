from ultralytics import YOLO

model = YOLO("best_latest.pt")  # Make sure best.pt is in the same folder

# Track with your MacBook's webcam
model.track(
    source=0,  # Webcam (built-in)
    tracker="botsort.yaml",  # Bult-in tracker
    show=True,  # Show live output
    save=False,  # Save to runs/track/predict/
    conf=0.7,  # Confidence threshold
)
