import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from anchors import AnchorParameters, anchors_for_shape
from utils.compute_overlap import compute_overlap


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

image_size = 512
# 512 (644, 49104) < 0.5 131
# 768 (644, 110484) < 0.5 63
# 896 (644, 150381) < 0.5 42
anchor_params = AnchorParameters()
anchors = anchors_for_shape((image_size, image_size), pyramid_levels=[3, 4, 5, 6, 7], anchor_params=anchor_params)

root = '/Users/yanyan/data/标注图片/'
properties = ['yes', 'no', 'unrecognized']
with open('car.txt', 'r') as f:
    img_lines = f.readlines()
save_path = '/Users/yanyan/data/plot_img_0'
if not os.path.exists(save_path):
    os.makedirs(save_path)


bbox = []
for img_line in img_lines:
    img_info = img_line.strip('\n').split('\t')
    img_path = img_info[0]

    img_boxes = img_info[1:]
    img = cv2.imread(root + img_path)
    img, scale = preprocess_image(img, image_size)
    for img_box in img_boxes:
        box = list(map(int, img_box.split(',')[:-2]))
        bbox.append(box)

bbox_array = np.array(bbox) * scale
# (644, 49104)
overlaps = compute_overlap(bbox_array.astype(np.float64), anchors.astype(np.float64))
# overlaps = compute_overlap(bbox_array.astype(anchors.astype(np.float64), np.float64))

print(overlaps.shape)
argmax_overlaps_inds = np.argmax(overlaps, axis=1)
max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
ingore = max_overlaps < 0.5

print(sum(ingore))

