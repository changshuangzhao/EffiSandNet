import cv2
import os
import numpy as np


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
    # image = image.astype(np.float32)
    # image /= 255.
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # image -= mean
    # image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale


root = '/Users/yanyan/data'
properties = ['yes', 'no', 'unrecognized']
with open('total_data.txt', 'r') as f:
    img_lines = f.readlines()

save_path = '/Users/yanyan/data/plot_img_0'
if not os.path.exists(save_path):
    os.makedirs(save_path)

min_bbox_w = 100000
min_bbox_h = 100000

yes = 0
no = 0
un = 0
for img_line in img_lines:
    img_info = img_line.strip('\n').split('\t')
    img_path = img_info[0]

    img_boxes = img_info[1:]

    img = cv2.imread(root + img_path)

    for img_box in img_boxes:
        box = list(map(int, img_box.split(',')[:-2]))
        # 求取gt中最小的宽，高
        # w = box[2] - box[0]
        # if w < min_bbox_w:
        #     min_bbox_w = w
        # h = box[3] - box[1]
        # if h < min_bbox_h:
        #     min_bbox_h = h

        # print('source box', box)
        # print('w = ', box[2] - box[0])
        # print('h = ', box[3] - box[1])
        # cls = img_box.split(',')[-2]
        pro = img_box.split(',')[-1]
        if pro == '0':
            yes += 1
        elif pro == '1':
            no += 1
        else:
            un += 1

        # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        # string = properties[int(pro)]
        # cv2.putText(img, string, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # img_copy = img.copy()
        # img_resize, scale = preprocess_image(img_copy, 768)
        # print(scale)
        # print(1 / scale)
        # box_resize = [int(i * scale) for i in box]
        # print('resize box', box_resize)
        # print('w = ', box_resize[2] - box_resize[0])
        # print('h = ', box_resize[3] - box_resize[1])
        # cv2.rectangle(img_resize, (box_resize[0], box_resize[1]), (box_resize[2], box_resize[3]), (0, 0, 255), 1)
        # string = properties[int(pro)]
        # cv2.putText(img_resize, string, (box_resize[0], box_resize[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # img_copy1 = img_copy.copy()
        # upsample_32_img = cv2.resize(img_copy1, (512 // 32, 512 // 32))
        # upsample_32_box = [int(i // 32) for i in box_resize]
        # print('upsample_32_box', upsample_32_box)
        # print('w = ', upsample_32_box[2] - upsample_32_box[0])
        # print('h = ', upsample_32_box[3] - upsample_32_box[1])
        # print()
        # cv2.rectangle(upsample_32_img, (upsample_32_box[0], upsample_32_box[1]), (upsample_32_box[2], upsample_32_box[3]), (0, 0, 255), 1)
        # string = properties[int(pro)]
        # cv2.putText(upsample_32_img, string, (upsample_32_box[0], upsample_32_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # print(os.path.join(save_path, os.path.basename(img_path)))
    # cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)
    # cv2.imshow('img', img)
    # cv2.imshow('resize img', img_resize)
    # cv2.imshow('upsample_32_box', upsample_32_img)

    # if cv2.waitKey() == 27:
    #     exit()
print(yes)
print(no)
print(un)
