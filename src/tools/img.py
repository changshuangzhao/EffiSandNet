import random
from easydict import EasyDict
import numpy as np
import cv2
import os


def cropimg(image, gt_box):
    # 1080 1920
    h, w = image.shape[:2]
    if h < cfgs.IMGHeight or w < cfgs.IMGWidth:
        return image, gt_box
    else:
        for i in range(100):
            dh, dw = int(random.random() * (h - cfgs.IMGHeight)), int(random.random() * (w - cfgs.IMGWidth))
            nx1 = dw
            nx2 = dw + cfgs.IMGWidth
            ny1 = dh
            ny2 = dh + cfgs.IMGHeight
            img = image[ny1:ny2, nx1:nx2, :]
            # gt = gt[dh:(dh+cfgs.IMGHeight),dw:(dw+cfgs.IMGWidth)]
            gt = gt_box.copy()
            keep_idx = np.where(gt[:, 2] > nx1)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:, 0] < nx2)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:, 3] > ny1)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:, 1] < ny2)
            gt = gt[keep_idx]
            gt[:, 0] = np.clip(gt[:, 0], nx1, nx2) - nx1
            gt[:, 2] = np.clip(gt[:, 2], nx1, nx2) - nx1
            gt[:, 1] = np.clip(gt[:, 1], ny1, ny2) - ny1
            gt[:, 3] = np.clip(gt[:, 3], ny1, ny2) - ny1
            # gt[:,0] = gt[:,0] / float(cfgs.IMGWidth)
            # gt[:,2] = gt[:,2] / float(cfgs.IMGWidth)
            # gt[:,1] = gt[:,1] / float(cfgs.IMGHeight)
            # gt[:,3] = gt[:,3] / float(cfgs.IMGHeight)
            gt_list = []
            for g in gt:
                if g[2] - g[0] >= 60 and g[3] - g[1] >= 60:
                    gt_list.append(g)
            gt_array = np.array(gt_list)
            if len(gt_array) > 0:
                break
        if len(gt_array) > 0:
            return img, gt_array
        return image, gt_box


# def normal(data):
#     _range = np.max(data) - np.min(data)
#     data = (data - np.min(data)) / _range
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     return np.divide(np.subtract(data, mean), std)
    # return data

if __name__ == '__main__':
    cfgs = EasyDict()
    cfgs.IMGWidth = 896
    cfgs.IMGHeight = 896
    root = '/Users/yanyan/data'
    with open('data/car.txt') as f:
        img_lines = f.readlines()

    for img_line in img_lines:
        img_info = img_line.strip('\n').split('\t')
        img_path = img_info[0]

        img_boxes = img_info[1:]
        img = cv2.imread(os.path.join(root, img_path))

        box_list = []
        for img_box in img_boxes:
            box = list(map(int, img_box.split(',')[:-2]))
            # 求取gt中最小的宽，高
            box_list.append(box)
        box_array = np.array(box_list)
        image, gt = cropimg(img, box_array)
        # image = normal(image)

        for i in gt:
            cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 2)
        cv2.imshow('img', image)
        cv2.waitKey()