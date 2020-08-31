import json
import os
import random

car_class = ['小轿车', '面包车', '货车', '小卡车', '大卡车', 'SUV', '客车', '公交车', '出租车', '垃圾车', '汽油罐车', '渣土车', '工程车', '厢式卡车', '洒水车', '水泥罐车', '其他']
car_property = ['苫盖', '未苫盖', '无法判断']
bbox_class = ['sand']
bbox_property = ['yes', 'no', 'unrecognized']
img_root = os.path.expanduser('~/data/SandCar/images')
anno_root = os.path.expanduser('~/data/SandCar/annotations')
anno_files = os.listdir(anno_root)


data_dict = {}  # {'img_path': [[x1,y1,x2,y2,cls,pro],[x1,y1,x2,y2,cls,pro]]
for anno_file in anno_files:
    if '.DS_Store' in anno_file:
        continue
    anno_path = os.path.join(anno_root, anno_file)
    print(anno_path)
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


val_csv = open('val.csv', 'w')
train_csv = open('train.csv', 'w')

index = 0
for key, values in data_dict.items():
    for value in values:
        left = value[0]
        top = value[1]
        right = value[2]
        bottom = value[3]
        cls = value[4]
        pro = value[5]
        img_path = os.path.join('SandCar/images', key)
        bbox_info = img_path + ',' + ','.join(list(map(str, [left, top, right, bottom]))) + ',' + cls + ',' + pro
        if index <= len(data_dict) * 0.1:
            val_csv.write(bbox_info + '\n')
        else:
            train_csv.write(bbox_info + '\n')
    index += 1
train_csv.close()
val_csv.close()

with open('cls.csv', 'w') as class_csv:
    class_csv.write('sand,0')
with open('pro.csv', 'w') as property_csv:
    property_csv.write('yes,0\nno,1\nunrecognized,2')


