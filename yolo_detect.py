import torch
import cv2

# Load YOLOv5s model (first time it will download weights)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    results.render()

    output = results.ims[0]

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv5 Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
