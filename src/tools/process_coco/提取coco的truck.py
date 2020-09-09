import os
import cv2
from shutil import copyfile
root = 'out_file.txt'
path = '/Users/yanyan/data/COCO2017/images'
save_path = '/Users/yanyan/data/coco_truck'
save_img = 0
no_img = 0

with open(root, 'r') as f:
    img_lines = f.readlines()

for img_line in img_lines:
    img_name = img_line.strip('\n').split(',')[0]
    name = img_name.split('/')[1]
    img_path = os.path.join(path, img_name)
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        copyfile(img_path, save_path + f'/{name}')
        print('保存图片', img_path)
        save_img += 1
    except FileNotFoundError:
        print('没有对应图片', img_path)
        no_img += 1
print(save_img)
print(no_img)


