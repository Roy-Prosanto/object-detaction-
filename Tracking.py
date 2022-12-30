import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

video = cv2.VideoCapture("Highway.mp4") # hear  you can change video

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = video.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[300: 720,450: 800]

    # 1. Object Detection
    black_mask = object_detector.apply(roi)
    _, mask = cv2.threshold(black_mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        #  find out the object and remove small object
        areaes = cv2.contourArea(cnt)
        if areaes >150:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # find  objectes
    object_boxes = tracker.update(detections)
    for object_boxes in object_boxes:
        x, y, w, h, id = object_boxes
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()