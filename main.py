"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# MQTT server environment variables
import os
import sys
import time
import socket
import json
import cv2 as cv
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
import numpy as np
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str,
                        default='person-detection-retail-0013.xml', help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, default='resources/Pedestrian_Detect_2_1_1.mp4', type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### DONE: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    dt = {}

    # Initialise the class
    Netw = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # variables for holding argument values
    model = args.model
    inp_vid = args.input
    ext = args.cpu_extension
    dev_used = args.device

    # extra variable for use
    start_time = 0
    creq_id = 0
    last_count = 0
    total_count = 0
    ### TODO: Load the model through `infer_network` ###

    n, c, h, w = Netw.load_model(dev_used, model, creq_id, ext)[1]

    imf = False  # flag to determine if input is image or video

    ### TODO: Handle the input stream ###

    if(inp_vid == '0'):
        stre = 0
    elif(inp_vid[:-4] == '.png' or inp_vid[:-4] == '.jpg' or inp_vid[:-4] == '.bmp' or inp_vid[:-4] == 'jpeg'):
        imf = True  # it is a image
        stre = inp_vid
    else:
        stre = inp_vid

    try:
        cap = cv.VideoCapture(stre)
    except(FileNotFoundError):
        print("Maybe wrong input path")
    except(Exception):
        print("Something went wrong ", Exception)

#     print(cap, '--')

    # Needed

    ### DONE: Loop until stream is over ###

    wide_c = cap.get(3)
    height_c = cap.get(4)
    probs = args.prob_threshold
    p_count = 0
    tcount = 0
    tf = 0
    while(cap.isOpened()):
        tf += 1
        k = cap.read()
        if not k[0]:
            break
        pi = k[1]
        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###
        im_frame = cv.resize(k[1], (w, h))
        im_frame = im_frame.transpose((2, 0, 1))
        im_frame = im_frame.reshape((n, c, h, w))
        ### TODO: Start asynchronous inference for specified request ###

        inf_init = time.time()
        Netw.exec_net(creq_id, im_frame)

        ### DONE: Wait for the result ###
#         print('Hello')

        if(Netw.wait(creq_id) == 0):
            inf_det = time.time()-inf_init
            inf_det = float(int(100000*inf_det))/100
            cv.putText(pi, "Inference Time : "+str(inf_det)+"ms", (25, 50),
                       cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
            ### DONE: Get the results of the inference request ###
            results = Netw.get_output(creq_id)
            c_count = 0
            for x in results[0][0]:
                t_v1 = 0
                t_v2 = 0
                if(x[2] >= prob_threshold):
                    xmin = int(x[3] * wide_c)
                    t_v1 = xmin
                    ymin = int(x[4] * height_c)
                    t_v2 = ymin
                    xmax = int(x[5] * wide_c)
                    ymax = int(x[6] * height_c)
                    cv.rectangle(pi, (xmin, ymin),
                                 (xmax, ymax), (0, 0, 255), 1)
                    c_count += 1
                cv.putText(pi, "Human "+str(x[2]), (t_v1, t_v2 - 7),
                           cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                cv.putText(pi, "Current Count : "+str(c_count), (25, 25),
                           cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

            client.publish("person", json.dumps(
                {"count": c_count}))  # People Count
#             client.publish("person", json.dumps({"total": total_count}))
#             log.basicConfig(level=os.environ.get("LOGLEVEL", dt))
            if(c_count >= p_count):
                dt[c_count] = time.time()
                tf = 0

                p_count = c_count
            else:
                if(tf == 4):  # Frame Buffer
                    u = p_count-c_count
                    total_count += u
                    pre = str(dt)
                    t_m = int(time.time()-dt[p_count-1])
                    t_m *= 1.78  # offset
                    client.publish("person/duration",
                                   json.dumps({"duration": t_m}))

                    p_count = c_count
                    c_count = 0


#                 client.publish("person", json.dumps({"total": total_count})) This is actually useless. This key doesn't make a difference

            ### DONE: Extract any desired stats from the results ###
#         sys.stdout.buffer.write( np.ascontiguousarray(pi, dtype=np.float32)) this is giving very strange results
        sys.stdout.buffer.write(pi)
        sys.stdout.flush()

        ### DONE: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        # Topic "person": keys of "count" and "total" ###  this DO NOT work
        ### Topic "person/duration": key of "duration" ###

        ### DONE: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
    cap.release()
    cv.destroyAllWindows()
    client.disconnect()
    # cleaning
    del Netw.net_plugin
    del Netw.network


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
