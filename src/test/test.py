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
    K.set_learning_phase(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # model_path = '../tools/cross5_visual/csv_62_0.2415_0.9798.h5'
    # model_path = '../tools/EffiSandCar/ep80-loss0.1919-val_loss0.8347.h5'
    model_path = '../train/models/ep01-loss2.4603-val_loss3.8886.h5'
    image_size = 896

    classes = {0: 'sand'}
    properties = {0: 'yes', 1: 'no', 2: 'unrecognized'}
    num_classes = 1
    num_properties = 3
    score_threshold = 0.5
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

    model, prediction_model = efficientdet(9, 1, 3, 64, 3, 3, 0.3, 0.5)
    prediction_model.load_weights(model_path, by_name=True)

    anchor_params = AnchorParameters()
    anchors = anchors_for_shape((image_size, image_size),
                                pyramid_levels=[3, 4, 5, 6, 7],
                                anchor_params=anchor_params)

    option = 'image'  # image | video
    if option == 'image':
        save_path = os.path.join(os.path.expanduser('~/data'), 'other_car_in_sand')
        with open('../generators/convert_data/val_cross0.csv', 'r') as f:
            img_infos = f.readlines()
        root = os.path.expanduser('~/data')
        for img_info in img_infos:
            img_info_split = img_info.strip('\n').split(',')
            img_name = img_info_split[0]

            img_path = os.path.join(root, img_name)

        # root = '/Users/yanyan/data/voc_bus'
        # img_names = os.listdir(root)
        # for img_name in img_names:
        #     img_path = os.path.join(root, img_name)
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
            if boxes.shape[0] != 0:
                # src_path = os.path.join(save_path, 'src_img')
                # plot_path = os.path.join(save_path, 'plot_img')
                # if not os.path.exists(src_path):
                #     os.makedirs(src_path)
                # if not os.path.exists(plot_path):
                #     os.makedirs(plot_path)
                # print(src_path + f"/{''.join(img_name.split('/')[-1]).split('.')[0]}.jpg", src_image)
                # exit()
                # cv2.imwrite(src_path + f"/{''.join(img_name.split('/')[-1]).split('.')[0]}.jpg", src_image)
                draw_boxes(src_image, boxes, scores, labels, pro_id, colors, classes, properties)
                # cv2.imwrite(plot_path + f"/{''.join(img_name.split('/')[-1]).split('.')[0]}.jpg", src_image)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', src_image)
            cv2.waitKey(0)
    elif option == 'video':
        root = '/Users/yanyan/Desktop/test_video'
        save_path = os.path.join(os.path.expanduser('~/data'), 'crop_img')
        video_names = os.listdir(root)
        for video_name in video_names:
            video_path = os.path.join(root, video_name)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("++++++++++++++++don't play++++++++++++++++++")
            else:
                total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print(f"++++++++++++++++total frame{total_frame}+++++++++++++++")
                frame_index = 0
                while frame_index < total_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, image = cap.read()
                    if ret:
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

                        if boxes.shape[0] != 0:
                            src_path = os.path.join(save_path, 'src_img')
                            plot_path = os.path.join(save_path, 'plot_img')
                            if not os.path.exists(src_path):
                                os.makedirs(src_path)
                            if not os.path.exists(plot_path):
                                os.makedirs(plot_path)

                            cv2.imwrite(src_path + f'/{video_name}_{frame_index}.jpg', src_image)
                            draw_boxes(src_image, boxes, scores, labels, pro_id, colors, classes, properties)
                            cv2.imwrite(plot_path + f'/{video_name}_{frame_index}.jpg', src_image)
                        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        # cv2.imshow('image', src_image)
                        # cv2.waitKey(25)
                    frame_index += 1
                cap.release()


if __name__ == '__main__':
    main()
