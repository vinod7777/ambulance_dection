import cv2
import torch
import warnings
import simpleaudio as sa

warnings.filterwarnings("ignore", category=FutureWarning)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
COCO_CLASSES = model.names


AMBULANCE_CLASSES = [ 'truck']
ALERT_SOUND = "alert.wav" 

def play_alert():
    try:
        wave_obj = sa.WaveObject.from_wave_file(ALERT_SOUND)
        play_obj = wave_obj.play()
    except Exception as e:
        print(f"Could not play sound: {e}")


cap = cv2.VideoCapture(0)
print("Starting detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    detected = False
    for label in labels:
        class_name = COCO_CLASSES[int(label)]
        if class_name in AMBULANCE_CLASSES:
            detected = True
            break

    if detected:
        print("ALERT: Ambulance detected!")
        play_alert()

    cv2.imshow('Ambulance Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 