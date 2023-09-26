import cv2
from tracker import *

# Create tracker object
tracker_ = EuclideanDistTracker()

capture = cv2.VideoCapture("dataset/반포대교-한남대교1080.mov")

# Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=750, varThreshold=500)

while True:
    ret, frame = capture.read()     # ret : True/False      frame : image
    # height, width, channel = frame.shape
    # print(height, width)

    # Extract Region of interest
    region_of_interest = frame[650:850,200:1000]

    # object detection on Jamsil direction area what I want
    mask = object_detector.apply(image=region_of_interest)

    _, mask = cv2.threshold(src=mask, thresh=128, maxval=255, type=cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(contour=contour)
        if area > 50 ** 2:
            # cv2.drawContours(image=region_of_interest, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=2)
            x, y, w, h = cv2.boundingRect(contour)


            detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker_.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(img=region_of_interest, text=str(id), org=(x, y-15), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.rectangle(img=region_of_interest, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)

    cv2.imshow(winname='frame', mat=frame)      # original frame
    cv2.imshow(winname="RoI", mat=region_of_interest)       # cropped frame
    cv2.imshow(winname='mask', mat=mask)        # Mixture of Gaussianed frame

    key = cv2.waitKey(delay=1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()