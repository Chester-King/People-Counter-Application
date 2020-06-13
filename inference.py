#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### DONE: Initialize any class variables desired ###
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.plugin = IECore()

        self.infer_request_handle = None

    def load_model(self, dev_used, mod_xml, num_req, ext=None):

        mod_bin = mod_xml[:-4] + ".bin"  # replacing .xml with .bin
#         print(ext)
        if ext and 'CPU' in dev_used:
            self.plugin.add_extension(ext, "CPU")

        ### DONE: Load the model ###

        self.network = IENetwork(model=mod_xml, weights=mod_bin)

        ### DONE: Check for supported layers ###

        if "CPU" in dev_used:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [
                lay for lay in self.network.layers.keys() if lay not in supported_layers]
            if len(not_supported_layers) != 0:
                sys.exit(1)

        ### DONE: Add any necessary extensions ###

        if ext and "CPU" in dev_used:
            self.plugin.add_extension(ext, "CPU")

        ### TODO: Return the loaded inference plugin ###
        if(num_req == 0):
            self.net_plugin = self.plugin.load_network(
                network=self.network, device_name=dev_used)
        else:
            self.net_plugin = self.plugin.load_network(
                network=self.network, device_name=dev_used, num_requests=num_req)

        ### DONE: Note: You may need to update the function parameters. ###

        return(self.plugin, self.get_input_shape())

    def get_input_shape(self):
        ### DONE: Return the shape of the input layer ###

        self.input_blob = next(iter(self.network.inputs))
        sh = self.network.inputs[self.input_blob].shape
        return(sh)

    def exec_net(self, creq_id, frame):
        self.infer_request_handle = self.net_plugin.start_async(
            request_id=creq_id, inputs={self.input_blob: frame})
        ### DONE: Start an asynchronous request ###
        return(self.net_plugin)

    def wait(self, creq_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        i_state = self.net_plugin.requests[creq_id].wait(-1)
        return(i_state)
        ### Note: You may need to update the function parameters. ###
        return

    def get_output(self, creq_id):
        # TODO: Extract and return the output results

        ### Note: You may need to update the function parameters. ###

        return(self.net_plugin.requests[creq_id].outputs['detection_out'])
