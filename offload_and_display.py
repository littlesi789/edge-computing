######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time

from npsocket import SocketNumpyArray

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu


# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])


# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps_list = np.ones((8))*10


input_mean = 127.5
input_std = 127.5

# socket
sock_sender = SocketNumpyArray()
sock_sender.initialize_sender('localhost', 9999)

data_receiver = SocketNumpyArray()
data_receiver.initalize_receiver(8848)


vid_shape = data_receiver.receive_array()
print(vid_shape[0], vid_shape[1])

print('...')
img_start = time.time()
while(video.isOpened()):
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (vid_shape[0], vid_shape[1]))

    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    floating_model = True
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    print('trying to send')
    sock_sender.send_numpy_array(input_data)
    print("Complete send")

    boxes, classes, scores = data_receiver.receive_array()
    # boxes = buffer[0]
    # classes = buffer[1]
    # scores = buffer[2]

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.rectangle(frame, (1,int(imH)-30), (1+450,int(imH)), (255, 255, 255), cv2.FILLED)

    check_time = time.time()
    latency = (check_time - img_start)
    img_start = check_time
    fps = (1/latency)
    fps_list = np.append(fps_list,fps)
    fps_list=np.delete(fps_list,0,0)
    # print(fps_list)
    cv2.putText(frame, "latency: {:.2f}ms mv_avg_fps: {}".format(latency*1000,  np.average(fps_list).astype('int')  ), (8,int(imH)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break


# Clean up
video.release()
cv2.destroyAllWindows()
sock_sender.send_numpy_array(np.array([999888]))
sock_sender.endServer()
data_receiver.endServer()