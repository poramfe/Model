import cv2
import mediapipe as mp
import time

#Load Face Detector
face_detection = mp.solutions.face_detection.FaceDetection()

# activate laptop camera
capture = cv2.VideoCapture(1)

# track TIME
starting_time = time.time()

while True:
    # return_value : True/False     frame : image array
    return_value, frame = capture.read()

    frame = cv2.flip(src=frame, flipCode=1)
    
    rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)
    print(results.detections)

    # Is there face DETECTED?
    if results.detections:
        elapsed_time = time.time() - starting_time
        # Draw elapsed time on screen
        cv2.putText(img=frame, text=f"{elapsed_time} seconds", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 0, 255), thickness=3)

        print(f"Elapsed : {elapsed_time}")
        print("Face looking at the screen")
    
    else:
        print("No FACE")

    cv2.imshow(winname="selfie", mat=frame)

    key = cv2.waitKey(delay=1)

    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()