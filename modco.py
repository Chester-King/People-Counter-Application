import tensorflow as tf
from inference import Network
import cv2
import time


def test_from_frozen_graph(pb_file, img_cv2):
    img_cv2 = cv2.resize(img_cv2, (224, 224))
    img = img_cv2[:, :, [2, 1, 0]]

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as session:
        session.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        inference_start_time = time.time()
        outputs = session.run([session.graph.get_tensor_by_name('num_detections:0'),
                               session.graph.get_tensor_by_name(
                                   'detection_scores:0'),
                               session.graph.get_tensor_by_name(
                                   'detection_boxes:0'),
                               session.graph.get_tensor_by_name('detection_classes:0')],
                              feed_dict={
            'image_tensor:0': img.reshape(1,
                                          img.shape[0],
                                          img.shape[1], 3)})
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time
        confidence = outputs[1][0][0]
        detection = outputs[2][0][0]
        return str(round(total_inference_time * 1000, 3)) + "ms", confidence


image = cv2.imread('resources/timage.jpg')

print("PERFORMANCE OF THE ORIGINAL MODEL")
print(test_from_frozen_graph(
    'model_2/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', image))
