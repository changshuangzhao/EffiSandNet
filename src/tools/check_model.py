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


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    val_generator = CSVGenerator(base_dir=cfg.DataRoot, data_file=cfg.ValData, class_file=cfg.Cls, property_file=cfg.Pro,
                                 batch_size=1, image_sizes=cfg.InputSize_w, shuffle_groups=False)

    num_classes = val_generator.num_classes()
    num_properties = val_generator.num_properties()
    num_anchors = val_generator.num_anchors

    model, prediction_model = efficientdet(num_anchors, num_classes, num_properties, cfg.w_bifpn, cfg.d_bifpn, cfg.d_head, score_threshold=0.01, nms_threshold=0.5)
    model.load_weights('../train/checkpoints/csv_212_1.1373_1.5797.h5', by_name=True)

    with open('../generators/convert_data/val.csv', 'r') as f:
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

        image_size = 896
        image, scale = preprocess_image(image, image_size=image_size)
        reg, cls, pro = model.predict_on_batch([np.expand_dims(image, axis=0)])
        print(reg)
