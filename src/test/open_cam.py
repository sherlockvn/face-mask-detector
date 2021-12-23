import numpy as np
import cv2
# configs
from src.config.config import PROJECT_PATH, CAFFE_PROTO_FILE, MODEL_FILE, CONFIDENCE
# video utils
import imutils, time
from imutils.video import VideoStream
# keras api
from tensorflow.python.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# load model
keras_model = load_model(PROJECT_PATH + "/models/model.h5")
# load predictive model from prototype 
# and trained model file caffenet which is for detecting human face
facenet = cv2.dnn.readNetFromCaffe(CAFFE_PROTO_FILE, MODEL_FILE)

# labels
label = {
    0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0}, # yellow text
    1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1}, # lightblue text
    2: {"name": "Without mask", "color": (0, 0, 255), "id": 2}, # red text
    3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3}, # green text
}

# open cam
cam = VideoStream(src=0).start()
time.sleep(2.0) 

# real time detection unless press 'q'
while True:
    # get frame from camera
    frame = cam.read()
    # resize the frame to width = 800
    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    facenet.setInput(blob)
    face = facenet.forward()

    # loop over the detections
    for i in range(0, face.shape[2]):
        confidence = face[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence < CONFIDENCE:
            continue

        # compute the coordinates of the bounding box of the object (x, y)
        face = face[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = face.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # extract the face ROI, ordering, resize it to 300x300, convert it from BGR to RGB channel, and preprocess it
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (300, 300))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        pred = np.argmax(keras_model.predict(face))

        # print(keras_model.predict(face))
        
        # label image
        classification = label[pred]["name"]
        color = (0,0,0)
        color = label[pred]["color"]

        # create rectangle inside the frame for label
        cv2.rectangle(frame, (startX,startY), (endX, endY), color, label[pred]["id"])

        # label image text
        cv2.putText(frame, classification, (startX, startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # faces detection
        cv2.putText(frame, f"{len(face)} detected face",(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Cam", frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xff == ord('q'):
        break