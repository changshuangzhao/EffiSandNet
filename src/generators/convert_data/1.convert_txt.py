import json
import os
import random

car_class = ['小轿车', '面包车', '货车', '小卡车', '大卡车', 'SUV', '客车', '公交车', '出租车', '垃圾车', '汽油罐车', '渣土车', '工程车', '厢式卡车', '洒水车', '水泥罐车', '其他']
car_property = ['苫盖', '未苫盖', '无法判断']
bbox_class = ['sand']
bbox_property = ['yes', 'no', 'unrecognized']
# img_root = os.path.expanduser('~/data/SandCar/images')
# anno_root = os.path.expanduser('~/data/SandCar/annotations')
img_root = os.path.expanduser('~/data')
anno_root = os.path.expanduser('~/data/object_detection/annotations')
anno_files = os.listdir(anno_root)


data_dict = {}  # {'img_path': [[x1,y1,x2,y2,cls,pro],[x1,y1,x2,y2,cls,pro]]
for anno_file in anno_files:
    if '.DS_Store' in anno_file:
        continue
    anno_path = os.path.join(anno_root, anno_file)
    # print(anno_path)
    with open(anno_path, 'r', encoding='utf8') as f:
        json_data = json.load(f)
        # 当前图像的路径信息
        img_path = json_data['data']['image_url'].split('/')[-1]
        img_infos = json_data['result']['data']

        # 遍历每一个bbox
        for index, img_info in enumerate(img_infos):
            bbox_label = img_info['label']
            bbox_category = img_info['category']
            category2label = {}
            for category, label in zip(bbox_category, bbox_label):
                category2label[category] = label

            if category2label.get('车辆标注-车辆类型', None) == '渣土车' and category2label.get('车辆标注-渣土车是否苫盖', None) is not None:
                # 当前bbox所对应的类别
                bbox_class_index = car_class.index(category2label.get('车辆标注-车辆类型', None))
                bbox_property_index = car_property.index(category2label.get('车辆标注-渣土车是否苫盖', None))
                # 当前bbox左上，右下坐标值
                bbox_4_point = img_info['coordinate']
                left = int(bbox_4_point[0]['x'])
                top = int(bbox_4_point[0]['y'])
                right = int(bbox_4_point[2]['x'])
                bottom = int(bbox_4_point[2]['y'])
                w = right - left
                h = bottom - top
                if min(w, h) <= 60:
                    continue
                # 当前框的info
                if data_dict.get(img_path) is None:
                    data_dict[img_path] = [[left, top, right, bottom, bbox_class[0], bbox_property[bbox_property_index]]]
                else:
                    data_dict[img_path].append([left, top, right, bottom, bbox_class[0], bbox_property[bbox_property_index]])
print('图片数量', len(data_dict))


data_list = []
for key, values in data_dict.items():
    for value in values:
        left = value[0]
        top = value[1]
        right = value[2]
        bottom = value[3]
        cls = value[4]
        pro = value[5]
        img_path = os.path.join('object_detection/images', key)
        bbox_info = img_path + ',' + ','.join(list(map(str, [left, top, right, bottom]))) + ',' + cls + ',' + pro
        data_list.append(bbox_info)


random.shuffle(data_list)
data_len = len(data_list)
val_csv_0 = open('val_cross0.csv', 'w')
train_csv_0 = open('train_cross0.csv', 'w')
val_csv_1 = open('val_cross1.csv', 'w')
train_csv_1 = open('train_cross1.csv', 'w')
val_csv_2 = open('val_cross2.csv', 'w')
train_csv_2 = open('train_cross2.csv', 'w')
val_csv_3 = open('val_cross3.csv', 'w')
train_csv_3 = open('train_cross3.csv', 'w')
val_csv_4 = open('val_cross4.csv', 'w')
train_csv_4 = open('train_cross4.csv', 'w')

for i, value in enumerate(data_list):
    if i < 294:
        val_csv_0.write(value + '\n')
    else:
        train_csv_0.write(value + '\n')

for i, value in enumerate(data_list):
    if i < 294:
        train_csv_1.write(value + '\n')
    elif i < 587:
        val_csv_1.write(value + '\n')
    else:
        train_csv_1.write(value + '\n')

for i, value in enumerate(data_list):
    if i < 587:
        train_csv_2.write(value + '\n')
    elif i < 879:
        val_csv_2.write(value + '\n')
    else:
        train_csv_2.write(value + '\n')

for i, value in enumerate(data_list):
    if i < 879:
        train_csv_3.write(value + '\n')
    elif i < 1173:
        val_csv_3.write(value + '\n')
    else:
        train_csv_3.write(value + '\n')

for i, value in enumerate(data_list):
    if i < 1173:
        train_csv_4.write(value + '\n')
    elif i < 1466:
        val_csv_4.write(value + '\n')
    else:
        train_csv_4.write(value + '\n')


with open('cls.csv', 'w') as class_csv:
    class_csv.write('sand,0')
with open('pro.csv', 'w') as property_csv:
    property_csv.write('yes,0\nno,1\nunrecognized,2')


