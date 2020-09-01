import cv2
import json
import numpy as np
import os
import time
import glob
from keras import backend as K
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from efficientdet import efficientdet
sys.path.append(os.path.dirname(__file__))
from utils import preprocess_image, postprocess_boxes, draw_boxes
from keras import models
sys.path.append(os.path.join(os.path.dirname(__file__), '../anchors'))
from anchor import AnchorParameters, anchors_for_shape


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    K.set_learning_phase(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_path = '../train/csv_15_1.7055_2.4735.h5'
    image_size = 896

    classes = {0: 'sand'}
    properties = {0: 'yes', 1: 'no', 2: 'unrecognized'}
    num_classes = 1
    num_properties = 3
    score_threshold = 0.25
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

    model, prediction_model = efficientdet(9, 1, 3, 64, 3, 3, 0.3, 0.5)
    prediction_model.load_weights(model_path, by_name=True)

    anchor_params = AnchorParameters()
    anchors = anchors_for_shape((image_size, image_size),
                                pyramid_levels=[3, 4, 5, 6, 7],
                                anchor_params=anchor_params)

    # for image_path in glob.glob('/Users/yanyan/data/SandCar/images/*.jpg'):
    with open('/Users/yanyan/Pictures/data_txt/val_3.csv', 'r') as f:
        img_infos = f.readlines()
    root = os.path.expanduser('~/data')
    for img_info in img_infos:
        img_info_split = img_info.strip('\n').split(',')
        img_name = img_info_split[0]
        # img_box = img_info_split[1:5]
        # img_clas = img_info_split[5]
        # img_pro = img_info_split[6]

        img_path = os.path.join(root, img_name)
        image = cv2.imread(img_path)

        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        # # [boxes, scores, labels, pro_id]
        boxes, scores, labels, pro_id = prediction_model.predict_on_batch([np.expand_dims(image, axis=0), np.expand_dims(anchors, axis=0)])
        boxes, scores, labels, pro_id = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(pro_id)
        print(time.time() - start)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]
        pro_id = pro_id[indices]
        draw_boxes(src_image, boxes, scores, labels, pro_id, colors, classes, properties)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', src_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
