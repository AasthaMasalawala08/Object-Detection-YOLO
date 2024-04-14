from ultralytics import YOLO
import cv2
import cvzone
import math


''' cap = cv2.VideoCapture(0)   # For Webcam
 cap.set(3, 1280)
cap.set(4, 720)'''

cap = cv2.VideoCapture("../Videos/cars.mp4")  # For video

model = YOLO("../Yolo-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "bike", "smart phone", "phone", "bus", "train", " truck", "boat",
              "traffic light", "fire wall", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "brocoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "cell phone",
              "microwave", "oven", "toaster ", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "cream", "pen", "boro plus",
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            # class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
