# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

After trying with 3 models finally person-detection-retail-0013 is being used for optimal output.

The command used to run the application is
`python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

Thus using the person-detection-retail-0013 model. Default probability is set to 0.6. This gives desired results.

## Explaining Custom Layers

There are diffrent processes for converting custom layers depending on the framework. For instance
The model [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

For converting Tensorflow model:

- Register custom layer as extension in model optimizer OR
- You need some sub graph that shoud not be in IR and also have another subgraph for that operation. Model Optimizer provides such solution as "Sub Graph Replacement" OR
- Pass the custom operation to Tensorflow to handle during inference.

The command used by me to convert this tensorflow model was `python python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

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

There were three models + Intel Zoo Model taken into account

[person-detection-retail-0013](https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/person-detection-retail-0013/FP32/)

| Parameters     | Stats    |
| -------------- | -------- |
| accuracy       | 0.92     |
| size           | 2.8 MB   |
| inference time | 46.03 ms |

[ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

| Parameters     | pre-conversion | post-conversion |
| -------------- | -------------- | --------------- |
| accuracy       | 0.8370055      | 0.7903          |
| size           | 105.83 MB      | 64.3 MB         |
| inference time | 3798.212 ms    | 76.5441 ms      |

[ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)

| Parameters     | pre-conversion | post-conversion |
| -------------- | -------------- | --------------- |
| accuracy       | 0.8770055      | 0.8236          |
| size           | 300.83 MB      | 95.4 MB         |
| inference time | 3593.265 ms    | 91.431 ms       |

[faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

| Parameters     | pre-conversion | post-conversion |
| -------------- | -------------- | --------------- |
| accuracy       | 0.9470055      | 0.9221          |
| size           | 200.83 MB      | 200 MB          |
| inference time | 5022.212 ms    | 4376.5441 ms    |

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

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

So I converted 3 models before switching to intel pretrained model which really seems to be the optimal option here due to following reasons.

The size of the converted model is still large compared to the intel pretrained model which is just around 3 MBs compared to the other converted model which was more than 100 MBs in general.

Another reason is the ssd_mobilenet_v2_coco and ssd_inception_v2_coco are not accurate enought when it comes to detecting humans. When executing the code with these models they totally skipped detecting of 1 human fully and also other human detection was not so accurate as the lost frame caused the calculation of total number of humans exceed way more than expected. Around 12.

As far as the faster_rcnn_inception_v2_coco model is concerned the accuracy is good enough but the inference time is around 5 seconds which defeats the purpose of realtime people counter and the size of the model didn't really change after the model optimization.

Thus using the [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) gaurentees correct results.
