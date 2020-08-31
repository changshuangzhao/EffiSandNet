import tensorflow as tf
import glob
import cv2
import numpy as np
import time
import os
import keras
import sys
sys.path.append(os.path.dirname(__file__))
from effdet_rpn_wrapper import cr_rpn_wrapper
import keras.backend as K


def read_pb_return_tensors(graph, pb_file, return_elements):
    with tf.gfile.GFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        in_out3 = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)
    return in_out3


def preprocess_image(image, image_size):
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale


def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def draw_boxes(image, boxes, scores, labels, pro_id, colors, classes, properties):
    for b, l, p, s in zip(boxes, labels, pro_id, scores):
        class_id = int(l)
        class_name = classes[class_id]

        property_id = int(p)
        property_name = properties[property_id]

        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.4f}'.format(s)
        color = colors[class_id]
        label = '-'.join([class_name, score, property_name])

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


class AnchorParameters:
    """
    The parameters that define how anchors are generated.

    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes=(32, 64, 128, 256, 512),
                 strides=(8, 16, 32, 64, 128),
                 ratios=(1, 0.5, 2),
                 scales=(2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=keras.backend.floatx())
        self.scales = np.array(scales, dtype=keras.backend.floatx())

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratio=h/w
    ratios=np.array([1, 0.5, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def guess_shapes(image_shape, pyramid_levels):
    """
    Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.

    Args:
        base_size:
        ratios:
        scales:

    Returns:

    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        anchor_params=None
):
    """
    Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    feature_map_shapes = guess_shapes(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4), dtype=np.float32)
    for idx, p in enumerate(pyramid_levels):
        # 每个尺度的9个基础anchor
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )


        # 每个尺度的anchor进行铺设
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)


def shift(feature_map_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    K.set_learning_phase(0)

    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(1)]
    classes_dict = {0: 'sand'}
    properties_dict = {0: 'yes', 1: 'no', 2: 'unrecognized'}

    return_elements = ['input_1:0', 'regression_1/concat:0', 'classification_1/concat:0', 'pro/concat:0']
    pb_file = 'pbfile/sand.pb'
    wrapper = cr_rpn_wrapper(pb_file, return_elements, score_t=0.1)

    img_path = 'cug_l_v1_f3475.jpg'
    image = cv2.imread(img_path)
    src_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    # 对图像进行填充
    image_size = 896
    image, scale = preprocess_image(image, image_size=image_size)

    anchor_params = AnchorParameters()
    anchors = anchors_for_shape((image_size, image_size), pyramid_levels=[3, 4, 5, 6, 7], anchor_params=anchor_params)
    final_boxes, final_scores, final_labels, final_pro_id = wrapper.predict_on_batch(image, anchors)
    final_boxes, final_scores, final_labels, final_pro_id = np.squeeze(final_boxes), np.squeeze(final_scores), np.squeeze(final_labels), np.squeeze(final_pro_id)
    # print(final_boxes)
    # print(final_scores)
    # print(final_labels)
    # print(final_pro_id)
    final_boxes = postprocess_boxes(boxes=final_boxes, scale=scale, height=h, width=w)

    final_score_threshold = 0.3
    indices = np.where(final_scores[:] > final_score_threshold)[0]
    final_boxes = final_boxes[indices]
    final_scores = final_scores[indices]
    final_labels = final_labels[indices]
    final_pro_id = final_pro_id[indices]

    draw_boxes(src_image, final_boxes, final_scores, final_labels, final_pro_id, colors, classes_dict, properties_dict)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', src_image)
    if cv2.waitKey() == 27:
        exit()
