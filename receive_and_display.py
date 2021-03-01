from npsocket import SocketNumpyArray

import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time

# socket
frame_receiver = SocketNumpyArray()
frame_receiver.initalize_receiver(9999)
time.sleep(1)
response_sender = SocketNumpyArray()
response_sender.initialize_sender('localhost', 8848)

try:
    while True:
        frame = frame_receiver.receive_array()
        if frame.shape == np.array([999888]).shape:
            print('end')
            cv2.destroyAllWindows()
            continue

        # Display
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        print('frame')

        response_sender.send_numpy_array(np.array([1]))
except:
    print("Exit server.")
    frame_receiver.endServer()
    response_sender.endServer()
    print("All socket shut down.")