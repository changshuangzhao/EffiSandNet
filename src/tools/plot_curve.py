import os
import sys
import tensorflow as tf
import keras.backend as K
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import progressbar

sys.path.append(os.path.join(os.path.dirname(__file__), '../generators'))
from generator import CSVGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from efficientdet import efficientdet
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '../eval'))
from visualization import draw_annotations, draw_detections
from colors import colors

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec = np.concatenate(([0.], rec, [1.]))
    prec = np.concatenate(([0.], prec, [0.]))
    # rec.insert(0, 0.0) # insert 0.0 at begining of list
    # rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    # prec.insert(0, 0.0) # insert 0.0 at begining of list
    # prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def mr_fppi(fppi, mr, score, threshold_list):
    mfppi = np.concatenate(([0.], fppi, [1.]))
    mmr = np.concatenate(([0.], mr, [0.]))
    for i in range(len(mmr) - 2, -1, -1):
        mmr[i] = max(mmr[i], mmr[i + 1])

    mr_list = []
    fppi_list = []
    for i, threshold in enumerate(threshold_list):
        index = np.greater(score, threshold)
        index = sum(index)

        mr_list.append(mr[index - 1])
        fppi_list.append(fppi[index - 1])

    return mmr, mfppi, mr_list, fppi_list


def plot_curve(generator, all_detections, all_annotations, save_path, iou_threshold=0.5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_tp = 0
    num_fp = 0
    sum_AP = 0
    ap_dictionary = {}
    color = [(0, 255, 0), (0, 0, 255)]
    # process detections and annotations
    # threshold of mr-fppi curve
    threshold_list = np.arange(0, 1, 0.02)
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            image = generator.load_image(i)
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
                    cv2.rectangle(image, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color=color[0], thickness=1)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    cv2.rectangle(image, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color=color[1], thickness=1)
            tp_fp = os.path.join(save_path, 'tf_fp')
            if not os.path.exists(tp_fp):
                os.makedirs(tp_fp)
            #cv2.imwrite(tp_fp + f'/{str(i)}.jpg', image)
        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            ap_dictionary[label] = 0.0
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

        average_precision, mrec, mprec = voc_ap(recall, precision)

        ap_dictionary[label] = average_precision
        sum_AP += average_precision

        # plot precision-recall curve
        plt.plot(recall, precision, '-o')
        # add a new penultimate point to the list (mrec[-2], 0.0)
        # since the last line segment (and respective area) do not affect the AP value
        area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
        area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
        # set window title
        fig = plt.gcf()  # gcf - get current figure
        fig.canvas.set_window_title('AP ' + str(label))
        # set plot title
        text = "{0:.2f}%".format(average_precision * 100) + " = " + str(label) + " AP "
        plt.title('class: ' + text)
        # plt.suptitle('This is a somewhat long figure title', fontsize=16)
        # set axis titles
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # optional - set axes
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        fig.savefig(os.path.join(save_path, str(label) + "_precision-recall.png"))
        plt.cla()  # clear axes for next plot

        # mr-fppi
        scores = scores[indices]
        fppi = 1 - precision
        mr = 1 - recall

        mmr, mfppi, mr_list, fppi_list = mr_fppi(fppi, mr, scores, threshold_list)

        # plot Mr-Fppi
        plt.plot(fppi, mr, '-o')
        area_under_curve_x = mfppi[:-1] + [mfppi[-2]] + [mfppi[-1]]
        area_under_curve_y = mmr[:-1] + [0.0] + [mmr[-1]]

        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
        # set window title
        fig = plt.gcf()  # gcf - get current figure
        fig.canvas.set_window_title('Mr-Fppi ' + str(label))
        plt.xlabel('FPPI')
        plt.ylabel('MR')
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        fig.savefig(os.path.join(save_path, str(label) + "_mr-fppi.png"))
        plt.cla()  # clear axes for next plot
        # plot Threshold-Fppi
        plt.plot(threshold_list, fppi_list, '-o')
        plt.fill_between(threshold_list, 0, fppi_list, alpha=0.2, edgecolor='r')
        fig = plt.gcf()  # gcf - get current figure
        fig.canvas.set_window_title('Threshold-Fppi ' + str(label))
        plt.xlabel('Threshold')
        plt.ylabel('FPPI')
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        fig.savefig(os.path.join(save_path, str(label) + "_threshold-fppi.png"))
        plt.cla()  # clear axes for next plot

        plt.plot(threshold_list, mr_list, '-o')
        plt.fill_between(threshold_list, 0, mr_list, alpha=0.2, edgecolor='r')
        # set window title
        fig = plt.gcf()  # gcf - get current figure
        fig.canvas.set_window_title('Threshold-Mr ' + str(label))
        # set axis titles
        plt.xlabel('Threshold')
        plt.ylabel('MR')
        # optional - set axes
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        fig.savefig(os.path.join(save_path, str(label) + "_threshold-mr.png"))
        plt.cla()  # clear axes for next plot

        # print(average_precision)
    mAP = sum_AP / generator.num_classes()
    print('num_fp', num_fp)
    print('num_tp', num_tp)
    print('mAP', mAP)


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
        detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if visualize:
            draw_annotations(src_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
                            colors=colors,
                            label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            cv2.imwrite(os.path.join('save_img', '{}.png'.format(i)), src_image)
            # cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            # cv2.imshow('{}'.format(i), src_image)
            # cv2.waitKey(0)

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
        save_path,
        # iou_threshold=0.5,
        # score_threshold=0.01,
        max_detections=100,
        visualize=False,
        epoch=0):
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
    # 大于0.5score的进行评估ap
    all_detections = _get_detections(generator, model, score_threshold=0.5, max_detections=max_detections, visualize=visualize)
    all_annotations = _get_annotations(generator)
    # plot_curve(generator, all_detections, all_annotations, save_path=save_path)


if __name__ == '__main__':
    val_generator = CSVGenerator(base_dir=cfg.DataRoot, data_file='../generators/convert_data/val.csv', class_file=cfg.Cls,
                                 property_file=cfg.Pro,
                                 batch_size=1, image_sizes=cfg.InputSize_w, shuffle_groups=False)

    num_classes = val_generator.num_classes()
    num_properties = val_generator.num_properties()
    num_anchors = val_generator.num_anchors

    model, prediction_model = efficientdet(num_anchors, num_classes, num_properties, cfg.w_bifpn, cfg.d_bifpn, cfg.d_head, score_threshold=0.5, nms_threshold=0.5)
    prediction_model.load_weights('../train/models/ep25-loss0.1936-val_loss0.9421.h5', by_name=True)
    evaluate(generator=val_generator, model=prediction_model, save_path='负例分析', visualize=True)