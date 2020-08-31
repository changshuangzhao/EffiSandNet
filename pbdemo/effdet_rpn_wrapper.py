import numpy as np
import time
import cv2
import tensorflow as tf


def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep


def default_bbox_transform_inv(boxes, deltas, scale_factors=None):
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = np.exp(tw) * wa
    h = np.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


class cr_rpn_wrapper:
    """
    cl refers to classification and regression
    """

    def __init__(self,
                 pb_file,
                 return_elements,
                 bbox_transform_inv_handler=None,
                 pyramid_level=(3, 4, 5),
                 iou_t=0.5,
                 score_t=0.1,
                 soft=False,
                 sigma=0.5,
                 class_specific_filter=True,
                 max_output_size=100):
        self.pb_file = pb_file
        self.return_elements = return_elements
        self.iou_t = iou_t
        self.score_t = score_t
        self.soft = soft
        self.sigma = sigma
        self.class_specific_filter = class_specific_filter
        self.max_output_size = max_output_size
        if bbox_transform_inv_handler is None:
            self.bbox_transform_inv = default_bbox_transform_inv
        self.pyramid_level = pyramid_level

        self.graph = tf.Graph()

    def read_pb_return_tensors(self, pb_file, return_elements):
        with tf.gfile.GFile(pb_file, 'rb') as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            in_out3 = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)
        return in_out3

    def predict_on_batch(self, img_input, anchors):
        # 填充后的高宽
        h, w = img_input.shape[:2]
        image = np.expand_dims(img_input, axis=0)

        sess = tf.Session(graph=self.graph)
        return_tensor = self.read_pb_return_tensors(pb_file=self.pb_file, return_elements=self.return_elements)
        regression, classification, pro = sess.run([return_tensor[1], return_tensor[2], return_tensor[3]], feed_dict={return_tensor[0]: image})

        bboxes = self.bbox_transform_inv(anchors, regression)
        bboxes = self.clip_box(bboxes, h, w)

        final_boxes, final_scores, final_labels, final_pro = [], [], [], []
        for n in range(1):
            classification_ = classification[n, ...]
            bboxes_ = bboxes[n, ...]
            pro_ = pro[n, ...]
            if self.class_specific_filter:
                all_indices = []
                for c in range(classification_.shape[1]):
                    scores = classification_[:, c]
                    labels = c * np.ones((scores.shape[0],), dtype='int64')
                    all_indices.append(self._filter_detections(scores, bboxes_, labels))
                indices = np.concatenate(all_indices, axis=1)
            else:
                scores = np.max(classification_, axis=0)
                labels = np.argmax(classification_, axis=0)
                indices = self._filter_detections(scores, bboxes_, labels)
            if indices.shape[-1] == 0:
                continue
            scores = classification_[indices[0], indices[1]]
            labels = indices[1]
            # select top k
            top_indices = np.argsort(scores)
            top_indices = top_indices[::-1][:np.minimum(self.max_output_size, scores.shape[0])]

            # filter input using the final set of indices
            indices = indices[0, top_indices]
            boxes = bboxes_[indices]
            pro_id = pro_[indices]
            pro_id = np.argmax(pro_id, axis=1)

            labels = labels[top_indices]
            scores = scores[top_indices]

            # zero pad the outputs
            pad_size = np.maximum(0, self.max_output_size - scores.shape[0])
            boxes = np.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1, mode='constant')
            scores = np.pad(scores, [[0, pad_size]], constant_values=-1, mode='constant')
            labels = np.pad(labels, [[0, pad_size]], constant_values=-1, mode='constant').astype('int32')
            pro_id = np.pad(pro_id, [[0, pad_size]], constant_values=-1, mode='constant').astype('int32')

            # set shapes, since we know what they are
            boxes.reshape([self.max_output_size, 4])
            scores.reshape([self.max_output_size])
            labels.reshape([self.max_output_size])
            pro_id.reshape([self.max_output_size])

            final_boxes.append(boxes[np.newaxis, ...])
            final_labels.append(labels[np.newaxis, ...])
            final_scores.append(scores[np.newaxis, ...])
            final_pro.append(pro_id[np.newaxis, ...])

        if len(final_scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return np.concatenate(final_boxes, axis=0), np.concatenate(final_scores, axis=0), np.concatenate(final_labels, axis=0), np.concatenate(final_pro, axis=0)

    def preprocess_image(self, image, image_size=512):
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

    def clip_box(self, boxes, height, width):
        x1 = np.clip(boxes[:, :, 0], 0, width - 1)
        y1 = np.clip(boxes[:, :, 1], 0, height - 1)
        x2 = np.clip(boxes[:, :, 2], 0, width - 1)
        y2 = np.clip(boxes[:, :, 3], 0, height - 1)
        return np.stack([x1, y1, x2, y2], axis=2)

    def _filter_detections(self, scores, bboxes, labels):
        # threshold based on score
        # (num_score_keeps, 1)
        indices = np.where(scores > self.score_t)[0]

        filtered_box = bboxes[indices]
        filtered_scores = scores[indices]

        nms_indices = py_cpu_softnms(filtered_box,
                              filtered_scores,
                              Nt=self.iou_t,
                              sigma=self.sigma,
                              thresh=self.score_t,
                              method=2 if self.soft else 3)
        keep = indices[nms_indices]

        indices_ = np.stack([keep, labels[keep]])
        return indices_


def test():
    from keras_module.modelzoo.yolo.yolo import get_v4slim
    import os, cv2
    yolo, yolo_p = get_v4slim(None,
                              '/Users/yuxunzhang/PycharmProjects/AlgoToolkit/keras_module/modelzoo/yolo/csdarknet53.cfg',
                              bbone_name='csdarknet53', #'csdarknet53',
                              bbone_weights=None,
                              num_channels=256,
                              with_dropblock=False)
    yolo.load_weights("/Users/yuxunzhang/PycharmProjects/AlgoToolkit/keras_module/modelzoo/yolo/checkpoints/coco_395_0.2898.h5")
    anchors_cfg = '8,13, 17, 26, 28, 52, 43, 107, 65, 58, 72, 195, 121, 123, 147, 290, 292, 404'
    anchor_params = FixedAnchorParameters(anchors_cfg, (2 ** 3, 2 ** 4, 2 ** 5))
    detector = cr_rpn_wrapper(model=yolo, anchor_params=anchor_params, score_t=0.00101, soft=False)

    image_path = '/Users/yuxunzhang/PycharmProjects/AlgoToolkit/keras_module/modelzoo/yolo/1.jpg'

    #est = np.random.rand(2, 608, 608, 3)
    boxes, scores, labels = detector.predict(test)







if __name__ == "__main__":
    test()