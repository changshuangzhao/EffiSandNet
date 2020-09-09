import os
import sys

root = '/Users/yanyan/data/object_detection/neg_hard'

image_names = os.listdir(root)
for image_name in image_names:
    if '.DS_Store' in image_name:
        continue
    with open('train.csv', 'a') as f:
        f.write('object_detection/neg_hard/' + image_name + ',' * 6 + '\n')

