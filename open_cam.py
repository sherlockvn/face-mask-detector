from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
from config import CAFFE_PROTO_FILE, MODEL_FILE, CONFIDENCE
import imutils, time
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


keras_model = load_model('model.h5')
facenet = cv2.dnn.readNetFromCaffe(CAFFE_PROTO_FILE, MODEL_FILE)

label = {
    0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0},
    1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1},
    2: {"name": "Without mask", "color": (0,0,255), "id": 2},
    3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3},
}

cam = VideoStream(src=0).start()
time.sleep(2.0) 

while True:
    frame = cam.read()
    # frame size
    frame = imutils.resize(frame, width=800)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and predictions
    facenet.setInput(blob)
    face = facenet.forward()
    # loop over the detections
    for i in range(0, face.shape[2]):
        confidence = face[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < CONFIDENCE:
            continue
        # compute the (x, y)-coordinates of the bounding box for the object
        face = face[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = face.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        # extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (300, 300))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        pred = np.argmax(keras_model.predict(face))
        # print(keras_model.predict(face))
        classification = label[pred]["name"]
        color = (0,0,0)
        color = label[pred]["color"]

        cv2.rectangle(frame, (startX,startY), (endX, endY), color, label[pred]["id"])

        cv2.putText(frame, classification, (startX, startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(face)} detected face",(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Cam", frame)
    # click Q to exits prediction
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    