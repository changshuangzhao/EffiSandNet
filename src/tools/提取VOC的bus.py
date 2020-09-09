import os
import cv2
from shutil import copyfile
root = '/Users/yanyan/data/VOC2012/ImageSets/Main/bus_test.txt'

with open(root, 'r') as f:
    img_lines = f.readlines()


img_list = []
for img_line in img_lines:
    img_split = img_line.strip('\n').split(' ')

    if len(img_split) == 3:
        if img_split[2] == '1':
            img_name = img_split[0]
            img_list.append(img_name)

path = '/Users/yanyan/data/VOC2012/JPEGImages'
save_img = 0
no_img = 0
for img_name in img_list:
    img_path = os.path.join(path, img_name + '.jpg')
    try:
        # if not os.path.exists(img_path):
        #     continue


        copyfile(img_path, f'/Users/yanyan/data/voc_bus/{img_name}' + '.jpg')
        print('保存图片', img_path)
        save_img += 1
    except FileNotFoundError:
        print('没有对应图片', img_path)
        no_img += 1
print(save_img)
print(no_img)
    # print(img_path)
    # img = cv2.imread(img_path)
    # # cv2.imwrite('')
    # cv2.imshow('img', img)
    # cv2.waitKey()
# print(img_list)

