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
    import keras
    import numpy as np
    import cv2

    sys.path.append(os.path.join(os.path.dirname(__file__), '../generators'))
    from generator import CSVGenerator

    sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
    from efficientdet import efficientdet

    sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
    from config import cfg

    def _bce(y_true, y_pred):
        pro_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        pro = y_pred

        indices = tf.where(keras.backend.equal(anchor_state, 1))
        pro = tf.gather_nd(pro, indices)
        pro_target = tf.gather_nd(pro_target, indices)

        # pro_loss = keras.losses.categorical_crossentropy(pro_target, pro, from_logits=True)
        # pro_loss = keras.losses.binary_crossentropy(pro_target, pro, from_logits=True)
        pro_loss = keras.backend.categorical_crossentropy(pro_target, pro, from_logits=False)
        # pro_loss = keras.backend.binary_crossentropy(pro_target, pro, from_logits=True)
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(pro_loss) / normalizer

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
    model.load_weights('../train/csv_72_1.4165_1.5192.h5', by_name=True)

    # anno reg, cls, pro
    for image, anno in val_generator:
        # print(anno)
        reg, cls, pro = model.predict_on_batch(image)

        reg_anchor_state = anno[0][:, :, -1]
        reg_indices = tf.where(keras.backend.equal(reg_anchor_state, 1))
        pos_reg = tf.gather_nd(reg, reg_indices)
        pos_tar_reg = tf.gather_nd(anno[0][:, :, :-1], reg_indices)

        cls_anchor_state = anno[1][:, :, -1]
        cls_indices = tf.where(keras.backend.not_equal(cls_anchor_state, -1))
        no_ign_labels = tf.gather_nd(cls, cls_indices)
        no_ign_tar_labels = tf.gather_nd(anno[1][:, :, :-1], cls_indices)

        pro_anchor_state = anno[2][:, :, -1]
        pro_indices = tf.where(keras.backend.equal(pro_anchor_state, 1))
        pos_pro = tf.gather_nd(pro, pro_indices)
        pos_tar_pro = tf.gather_nd(anno[2][:, :, :-1], pro_indices)
        # print()
        # print(reg.shape)
        # print(cls.shape)
        # print(pro.shape)
        # exit()
        # _bce(anno[2], pro)

