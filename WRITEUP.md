# Project Write-Up

Here is the project write up detailing important components of the project.

## Explaining Custom Layers

There are diffrent processes for converting custom layers depending on the framework.
The model used here is [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

For converting Tensorflow model:

- Register custom layer as extension in model optimizer OR
- You need some sub graph that shoud not be in IR and also have another subgraph for that operation. Model Optimizer provides such solution as "Sub Graph Replacement" OR
- Pass the custom operation to Tensorflow to handle during inference.

The command used by me to convert this tensorflow model was `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`

Apart from this you can also convert Caffe and MXNet Models as given below

For Caffe:

- Register custom layer as extension in model optimizer OR
- Register custom layer as Custom and use system's caffe to calculate the output shape of each Custom Layer

For MXNet:

- MXNet's process is same as tensorflow one. It only not supporting the offloading the custom layer to MXNet to handle.

The Custom layes known as per their name "Custom" means modified or new. There are variety of frameworks which are used for training the deep learning models such as Keras, Tensorflow, ONNX, Caffe etc.

All these frameworks have their own methods to process the tensors (Data) so it may possible that some functions are not available or behaves diffrently in each other.

Hence Custom Layer support neccessary for Model Optimizer so that the unsupported operations can be supported through dependent framework during runtime inference.

Model Optimizer query each layer of trained model from the list of known layers (Supported layers) before building the model's internal representation. It also optimizes the model by following three steps. Quantization, Freezing and Fusing. At last it generated the intermidiate representation from the trained model.

## Comparing Model Performance

Comparing Models:

- Comparing the size of both models
- Comparing the accuracy of models
- Comparing the inference time of both models

To get the original model stats [modco.py](./modco.py) is used
To get the stats of the optimized model stats were logged from the main.py

| Parameters     | pre-conversion | post-conversion |
| -------------- | -------------- | --------------- |
| accuracy       | 0.8370055      | 0.7903          |
| size           | 200.83 MB      | 64.3 MB         |
| inference time | 3798.212 ms    | 76.5441 ms      |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1. Event Attendance Statistics
2. Number of people still present in a supermarket
3. Social Distancing maintaninance
4. Security Systems

Each of these use cases would be useful because...

1. Event Attendance Statistics - Important to see how many poeple attending an event thus determining the popularity of the event.
2. Number of people still present in a shop - One at entery point and one at exit point will determine the number of people present in a shop in realtime.
3. Social Distancing maintaninance - Counting the number of people present in a restaurant and displaying it publically so that people only go to the restaurant if the number of people present in it are less than a safety threshold.
4. Security System - Using people counter to determine the number of people present in the driveway at any given moment of time.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- In bad light, model might not detect peple. We might have to use image filters on the input to make it more clear for the model to detect.
- Depending on the situation and need model accuracy comes into play. Most of the day to day needs no not reaquire highly accurate detection. However in some extreme cases like security of a High Profile personnel
- Camera focal length/image size may play an important factor. For instance if the target object is very far but due to the Camera Focal length it is zoomed in and clearly visible then the model can easily detect the person.
