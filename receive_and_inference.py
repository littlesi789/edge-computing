from npsocket import SocketNumpyArray

import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time
import signal

from tensorflow.lite.python.interpreter import Interpreter



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')

parser.add_argument('--grace_shutdown', help='set true to shut down socket when sending ctrl-C',
                    default=True)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

interpreter = Interpreter(model_path=PATH_TO_CKPT, num_threads=2)


interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


# socket
frame_receiver = SocketNumpyArray()
print("++++++ Ready ++++++")
frame_receiver.initalize_receiver(9999)
response_sender = SocketNumpyArray()
response_sender.initialize_sender('localhost', 8848)

if args.grace_shutdown:
    def signal_handler(sig, frame):
        print('Ctrl-C: Gracefully shut down the server.')
        print("Exit server.")
        try:
            frame_receiver.closeServer()
        except:
            pass
        try:
            response_sender.closeServer()
        except:
            pass
        print("All socket shut down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)



response_sender.send_numpy_array([height, width])
try:
    while True:
        frame = frame_receiver.receive_array()
        print(frame.shape)
        if frame.shape == np.array([999888]).shape:
            print('End of service.')
            frame_receiver.endServer()
            response_sender.endServer()
            print('All server closed.')
            frame_receiver = SocketNumpyArray()
            response_sender = SocketNumpyArray()
            frame_receiver.initalize_receiver(9999)
            response_sender.initialize_sender('localhost', 8848)
            response_sender.send_numpy_array([height, width])
            continue

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],frame)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        print(boxes.shape, classes.shape, scores.shape)

        response_sender.send_numpy_array( [boxes,classes,scores] )
        # response_sender.send_numpy_array( classes )
        # response_sender.send_numpy_array( scores )

        # response_sender.send_numpy_array(np.array([1]))
except Exception as e:
    print(e)
    print("Exit server.")
    frame_receiver.endServer()
    response_sender.endServer()
    print("All socket shut down.")