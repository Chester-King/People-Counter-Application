
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
import cv2
import math
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
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str, default="model_1/frozen_inference_graph.xml",
                        help="XML file path")
    parser.add_argument("-i", "--input", required=False, type=str, default="./resources/Pedestrian_Detect_2_1_1.mp4",
                        help="Input file path")
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
                        help="Probability threshold for detections filtering - default value 0.5")
    return parser


def connect_mqtt():
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def draw_bounding_boxs(coordinates, frame, wc, hc, x, k):
    # Draw the bounding box

    curr_count = 0
    dis = x
    for obj in coordinates[0][0]:

        if obj[2] > prob_threshold:
            xmin = int(obj[3] * wc)
            ymin = int(obj[4] * hc)
            xmax = int(obj[5] * wc)
            ymax = int(obj[6] * hc)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            curr_count += 1
            text = '{}, %: {}'.format("HUMAN", round(obj[2], 3))
            cv2.putText(frame, text, (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

            c_x = frame.shape[1]/2
            c_y = frame.shape[0]/2
            mid_x = (xmax + xmin)/2
            mid_y = (ymax + ymin)/2

            # Calculating distance
            dis = math.sqrt(math.pow(mid_x - c_x, 2) +
                            math.pow(mid_y - c_y, 2) * 1.0)
            k = 0

    if curr_count < 1:
        k += 1

    if dis > 0 and k < 10:
        curr_count = 1
        k += 1
        if k > 100:
            k = 0

    return frame, curr_count, dis, k


def infer_on_stream(args, client):
    # Initialise the class and assigning values to variables
    infer_network = Network()
    model = args.model
    video_file = args.input
    extn = args.cpu_extension
    device = args.device

    # Flag for the input image
    iflag = False

    start_time = 0
    current_request_id = 0
    last_count = 0
    total_count = 0

    # Load the model through `infer_network`
    n, c, h, w = infer_network.load_model(
        model, device, 1, 1, current_request_id, extn)[1]

    # Handle the input stream
    if video_file == 'CAM':  # live feed
        input_stream = 0

    elif video_file.endswith('.jpg') or video_file.endswith('.bmp'):    # image
        iflag = True
        input_stream = video_file

    else:  # Video
        input_stream = video_file

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Video file not present at the given location: " + video_file)
    except Exception as e:
        print("Something is wrong with the video file ", e)

    global wc, hc, prob_threshold
    total_count = 0
    duration = 0

    wc = cap.get(3)
    hc = cap.get(4)
    prob_threshold = args.prob_threshold
    temp = 0
    bbx = 0

    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Pre-process the image as needed

        # Preprocessing input
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        # Start asynchronous inference
        inf_start = time.time()
        infer_network.exec_net(current_request_id, image)

        # Wait for the result
        if infer_network.wait(current_request_id) == 0:
            det_time = time.time() - inf_start

            # Get the results of the inference request
            result = infer_network.get_output(current_request_id)

            # Draw Bounting Box
            frame, current_count, d, bbx = draw_bounding_boxs(
                result, frame, wc, hc, temp, bbx)

            # Displaying Inference Time
            inf_time_message = "Inference time: {:.3f}ms".format(
                det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)

            # Calculating information
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            if current_count < last_count:  # Average Time
                duration = int(time.time() - start_time)
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            # Displaying Distance and Current count
            text = "Distance: %d" % d
            cv2.putText(frame, text, (15, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)
            text = " Lost frame: %d" % bbx
            cv2.putText(frame, text, (15, 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)
            text = "Current count: %d " % current_count
            cv2.putText(frame, text, (15, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 155, 0), 1)

            if current_count > 3:
                txt2 = "Maximum count reached :)"
                (text_width, text_height) = cv2.getTextSize(
                    txt2, cv2.FONT_HERSHEY_COMPLEX, 0.5, thickness=1)[0]
                0

            client.publish("person", json.dumps(
                {"count": current_count}))  # People Count

            last_count = current_count
            temp = d

            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Save the Image
        if iflag:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


def main():
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
