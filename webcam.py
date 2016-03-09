import cv2
import sys
import numpy as np
import time

cascPath = sys.argv[1]
skip = 0
faceCascade = cv2.CascadeClassifier(cascPath)
faces = []
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    # skip +=1
    # if skip%4 == 0:
    # continue
  ret, frame = video_capture.read()
  if (ret is True) and (frame is not None): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    skip +=1
    if skip%3 == 0:
        faces = faceCascade.detectMultiScale(
           gray,
           scaleFactor=1.1,
           minNeighbors=5,
           minSize=(30, 30),
           flags=2
        )

        # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else:
    print('capture was empty')
    print(ret)
    print(frame is not None)
    time.sleep(1)	
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
