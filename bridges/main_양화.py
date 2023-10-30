"""양화대교 주간"""
""" one way """
import cv2
import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.centeroid_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rectangle):
        # Objects boxes and ids
        objects_bounding_boxes_ids = []

        # Get center point of new object
        for rectangle in objects_rectangle:
            x, y, width, height = rectangle;    centeriod_of_x = (x + x + width) // 2;    centeroid_of_y = (y + y + height) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, point in self.centeroid_points.items():
                distance = math.hypot(centeriod_of_x - point[0], centeroid_of_y - point[1])

                if distance < 25:
                    self.centeroid_points[id] = (centeriod_of_x, centeroid_of_y)
                    # print(self.centeroid_points)
                    objects_bounding_boxes_ids.append([x, y, width, height, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.centeroid_points[self.id_count] = (centeriod_of_x, centeroid_of_y)
                objects_bounding_boxes_ids.append([x, y, width, height, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for object_boundingbox_id in objects_bounding_boxes_ids:
            _, _, _, _, object_id = object_boundingbox_id
            center = self.centeroid_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.centeroid_points = new_center_points.copy()

        return objects_bounding_boxes_ids
    




upload_url = "http://localhost:4000/api/videos/upload"

capture = cv2.VideoCapture(filename="./bridges/dataset/양화대교/2023-10-10 13-37-09_TUEone.mov")
object_detector = cv2.createBackgroundSubtractorMOG2(history=450, varThreshold=32, detectShadows=True)
tracker = EuclideanDistTracker()

frame_counter = 0
total_id_10sec_ago = 0

while True:
    return_value, frame = capture.read()
    if return_value == False: break

    # print(frame.shape)
    frame_Region_of_Interest = frame[800:900, 550:1000]
    # frame_Region_of_Interest += 50

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)
    brightness, MoG_mask = cv2.threshold(src=MoG_mask, thresh=32, maxval=255, type=cv2.THRESH_BINARY)

    object_location = []

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= 50 ** 2:
            x, y, w, h = cv2.boundingRect(array=contour)
            object_location.append([x, y, w, h])
            # cv2.drawContours(image=frame_Region_of_Interest, contours=contour, contourIdx=-1, color=(255, 255, 255), thickness=5)

    boxes_ids = tracker.update(objects_rectangle=object_location)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(img=frame_Region_of_Interest, text=str(id), org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=3)
        cv2.rectangle(img=frame_Region_of_Interest, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=3)

    frame_counter += 1
    current_total_id = id
    if frame_counter % 3600 == 0:
        print(f"{(current_total_id - total_id_10sec_ago) / 1} cars per 1 minute")
        total_id_10sec_ago = id

    cv2.imshow(winname="YangHwa Bridge", mat=frame)
    # cv2.imshow(winname="mask", mat=MoG_mask)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()