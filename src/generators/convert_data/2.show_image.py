import cv2
import os

properties = ['yes', 'no', 'unrecognized']
classes = ['sand']


with open('val.csv', 'r') as f:
    img_info_list = f.readlines()

img_list = []
for img_info in img_info_list:
    img_info_line = img_info.strip('\n').split(',')
    img_name = img_info_line[0]
    img_box = list(map(int, img_info_line[1:5]))
    img_class = img_info_line[5]
    img_property = img_info_line[6]

    img_path = os.path.join(os.path.expanduser('~/data'), img_name)

    img = cv2.imread(img_path)
    cv2.rectangle(img, (img_box[0], img_box[1]), (img_box[2], img_box[3]), (0, 0, 255), 2)
    string = f'class={img_class}' + ' ' + f'property={img_property}'
    cv2.putText(img, string, (img_box[0] + 2, img_box[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)
    cv2.imshow('image', img)
    print(img_path)
    if cv2.waitKey() == 27:
        exit()



