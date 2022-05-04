import cv2
import numpy as np
import torch
from torch import hub # Hub contains other models like FasterRCNN
import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras import utils
from keras.utils.np_utils import to_categorical
import pandas as pd
import tensorflow as tf

#live recognition

# def plot_boxes(results, frame,model):
#   labels, cord = results
#   n = len(labels)
#
#   x_shape, y_shape = frame.shape[1], frame.shape[0]
#   for i in range(n):
#     row = cord[i]
#     # If score is less than 0.2 we avoid making a prediction.
#     if row[4] < 0.2:
#       continue
#     x1 = int(row[0] * x_shape)
#     y1 = int(row[1] * y_shape)
#     x2 = int(row[2] * x_shape)
#     y2 = int(row[3] * y_shape)
#     bgr = (0, 255, 0)  # color of the box
#     classes = model.names  # Get the name of label index
#     label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.
#     #print(classes[labels[i].astype(int)])
#     cv2.rectangle(frame, \
#                   (x1, y1), (x2, y2), \
#                   bgr, 2)  # Plot the boxes
#     cv2.putText(frame, \
#                 classes[labels[i].astype(int)], \
#                 (x1, y1), \
#                 label_font, 0.9, bgr, 2)  # Put a label over box.
#     return frame


# model = torch.hub.load( \
#                       'ultralytics/yolov5', \
#                       'yolov5s', \
#                       pretrained=True)

#model = keras.models.load_model("mod")
model = keras.models.load_model('mod2BoundingBox')
df = pd.read_csv("split-data/train_new.csv")
x = df['category'].values
classes = np.unique(x)

cap = cv2.VideoCapture(0) # 0 means read from local camera.




# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name


# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)

    eroded = cv2.erode(thresh,kernel)
    dilated = cv2.dilate(eroded,kernel)

    d2 = cv2.dilate(dilated,kernel)
    e2 = cv2.erode(d2,kernel)

    #cv2.imshow('Frame', frame)
    dim = (256, 256)
    f2 = cv2.resize(frame,dim)
    imlist = []
    imlist.append(f2)
    f3 = np.asarray(imlist)

    results = model.predict(f3)

    print("------------------------------")

    #print(classes[np.argmax(results[0], axis=1)])
    print(classes[np.argmax(results[0], axis=1)])

    tf = cv2.putText(frame,classes[np.argmax(results[0], axis=1)])
    cv2.imshow('Frame', tf)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break


  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


