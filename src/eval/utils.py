"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cv2
import progressbar

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import os
import sys
sys.path.append(os.path.dirname(__file__))
from visualization import draw_detections, draw_annotations


def compute_iou(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 4) anchors
    :param boxes_b: (M, 4) box
    :return: IOU (N, M)
    """
    boxes_a = boxes_a[:, np.newaxis, :]  # N, 1, 4
    boxes_b = boxes_b[np.newaxis, ...]  # 1, M, 4
    overlap_w = np.maximum(0.0,
                           np.minimum(boxes_a[..., 2], boxes_b[..., 2]) -
                           np.maximum(boxes_a[..., 0], boxes_b[..., 0]) + 1)
    overlap_h = np.maximum(0.0,
                           np.minimum(boxes_a[..., 3], boxes_b[..., 3]) -
                           np.maximum(boxes_a[..., 1], boxes_b[..., 1]) + 1)

    overlap = overlap_w * overlap_h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0] + 1) * (boxes_a[..., 3] - boxes_a[..., 1] + 1)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0] + 1) * (boxes_b[..., 3] - boxes_b[..., 1] + 1)

    iou = overlap / (area_a + area_b - overlap)
    return iou


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, visualize=False):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image = generator.load_image(i)
        src_image = image.copy()
        h, w = image.shape[:2]

        anchors = generator.anchors
        image, scale = generator.preprocess_image(image)
        # run network
        # boxes, scores, *_, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels, pro_id = model.predict_on_batch([np.expand_dims(image, axis=0),  # (1, 896, 896, 3)
                                                                np.expand_dims(anchors, axis=0)])  # (1, 150381, 4)
        boxes /= scale
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        # (n, 4)
        image_boxes = boxes[0, indices[scores_sort], :]
        # (n, )
        image_scores = scores[scores_sort]
        # (n, )
        image_labels = labels[0, indices[scores_sort]]
        # (n, )
        image_pro = pro_id[0, indices[scores_sort]]
        # (n, 6)
        detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if visualize:
            draw_annotations(src_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
                            label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            cv2.imshow('{}'.format(i), src_image)
            cv2.waitKey(0)

        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]

    return all_detections


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        visualize=False,
        epoch=0
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.

    Returns:
        A dict mapping class names to mAP scores.

    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, visualize=visualize)
    all_annotations = _get_annotations(generator)
    average_precisions = {}
    num_tp = 0
    num_fp = 0

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_iou(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        if false_positives.shape[0] == 0:
            num_fp += 0
        else:
            num_fp += false_positives[-1]
        if true_positives.shape[0] == 0:
            num_tp += 0
        else:
            num_tp += true_positives[-1]

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    print('num_fp={}, num_tp={}'.format(num_fp, num_tp))

    return average_precisions


if __name__ == '__main__':
    import os
    import sys
    import tensorflow as tf
    import keras.backend as K
    import time
    import numpy as np
    import cv2

    sys.path.append(os.path.join(os.path.dirname(__file__), '../generators'))
    from generator import CSVGenerator
    sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
    from efficientdet import efficientdet
    sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
    from config import cfg

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    val_generator = CSVGenerator(base_dir=cfg.DataRoot, data_file=cfg.ValData, class_file=cfg.Cls,
                                 property_file=cfg.Pro,
                                 batch_size=1, image_sizes=cfg.InputSize_w, shuffle_groups=False)

    num_classes = val_generator.num_classes()
    num_properties = val_generator.num_properties()
    num_anchors = val_generator.num_anchors

    model, prediction_model = efficientdet(num_anchors, num_classes, num_properties, cfg.w_bifpn, cfg.d_bifpn, cfg.d_head, score_threshold=0.01, nms_threshold=0.5)
    prediction_model.load_weights('../train/checkpoints/csv_212_1.1373_1.5797.h5', by_name=True)
    evaluate(generator=val_generator, model=prediction_model, iou_threshold=0.5, score_threshold=0.25)
