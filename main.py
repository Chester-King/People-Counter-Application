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
import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str,
                        help="Path to an xml file with a trained model.", default="model_2/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml")
    parser.add_argument("-i", "--input", required=False, type=str, default="./resources/Pedestrian_Detect_2_1_1.mp4",
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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering - default value 0.6")
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):

    # Initialise the class
    infer_network = Network()
    prob_threshold = args.prob_threshold
    model = args.model
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension

    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    # Checks for live feed
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input

    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)

    w = int(cap.get(3))
    h = int(cap.get(4))

    input_shape = network_shape['image_tensor']

    duration_previous = 0
    ct = 0
    duration = 0
    request_id = 0

    report = 0
    counter = 0
    counter_prev = 0

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)

        net_input = {'image_tensor': image_p, 'image_info': image_p.shape[1:]}
        duration_report = None
        inf_start = time.time()
        infer_network.exec_net(net_input, request_id)

        if infer_network.wait() == 0:
            det_time = time.time() - inf_start
            net_output = infer_network.get_output()
            pointer = 0
            probs = net_output[0, 0, :, 2]
            text = ""
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    pointer += 1
                    box = net_output[0, 0, i, 3:]
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                    text = '{}, %: {}'.format("HUMAN", round(p, 3))

            inf_time_message = "Inference time: {:.3f}ms".format(
                det_time * 1000)
            cv2.putText(frame, text, (int(
                box[0] * w), int(box[2] * w) - 7), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
            cv2.putText(frame, inf_time_message, (15, 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)
            if pointer != counter:
                counter_prev = counter
                counter = pointer
                if duration >= 3:
                    duration_previous = duration
                    duration = 0
                else:
                    duration = duration_previous + duration
                    duration_previous = 0  # unknown, not needed in this case
            else:
                duration += 1
                if duration >= 3:
                    report = counter
                    if duration == 3 and counter > counter_prev:
                        ct += counter - counter_prev
                    elif duration == 3 and counter < counter_prev:
                        duration_report = int(
                            (duration_previous / 10.0) * 1000)

            # current_count, total_count and duration to the MQTT server
            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': ct}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps(
                                   {'duration': duration_report}),
                               qos=0, retain=False)
            text = "Current count: %d " % report
            cv2.putText(frame, text, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)
            text = "Total count: %d " % ct
            cv2.putText(frame, text, (15, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)

        # Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()


def main():
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
