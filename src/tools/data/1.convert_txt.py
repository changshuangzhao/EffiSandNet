import json
import os

car_class = ['小轿车', '面包车', '货车', '小卡车', '大卡车', 'SUV', '客车', '公交车', '出租车', '垃圾车', '汽油罐车', '渣土车', '渣土车且小卡车', '工程车', '厢式卡车', '洒水车', '水泥罐车', '其他']
car_property = ['苫盖', '未苫盖', '无法判断']

anno_root = '/Users/yanyan/data/object_detection/annotations'
# anno_root = '/Users/yanyan/data/labelling_images/SandCar2/annotations'
anno_files = os.listdir(anno_root)
# car_txt = open('total_data.txt', 'w')
number = 0
for anno_file in anno_files:
    if '.DS_Store' in anno_file:
        continue
    anno_path = os.path.join(anno_root, anno_file)
    with open(anno_path, 'r', encoding='utf8') as f:
        json_data = json.load(f)
        # 当前图像的路径信息
        img_path = '/'.join(json_data['data']['image_url'].split('/')[-2:])

        img_name = img_path.split('/')[1]
        img_infos = json_data['result']['data']

        bbox_info = ''
        for index, img_info in enumerate(img_infos):
            bbox_label = img_info['label']
            bbox_category = img_info['category']
            category2label = {}
            for category, label in zip(bbox_category, bbox_label):
                category2label[category] = label

            if '渣土车' in str(category2label.get('车辆标注-车辆类型', None)) and category2label.get('车辆标注-渣土车是否苫盖', None) is not None:
                # 当前bbox所对应的类别
                bbox_class_index = car_class.index(category2label.get('车辆标注-车辆类型', None))
                bbox_property_index = car_property.index(category2label.get('车辆标注-渣土车是否苫盖', None))
                # 当前bbox左上，右下坐标值
                bbox_4_point = img_info['coordinate']
                left = int(bbox_4_point[0]['x'])
                top = int(bbox_4_point[0]['y'])
                right = int(bbox_4_point[2]['x'])
                bottom = int(bbox_4_point[2]['y'])
                # 当前框的info
                bbox_info += ','.join(list(map(str, [left, top, right, bottom, bbox_class_index, bbox_property_index]))) + '\t'
                number += 1
            elif category2label.get('车辆标注-车辆类型', None) == '渣土车且小卡车':
                print('渣土车且小卡车', img_path)
            elif category2label.get('车辆标注-车辆类型', None) == '渣土车':
                print('是渣土车，但是缺少属性信息', img_path)

        # if bbox_info:
        #     car_txt.write('object_detection/images/' + img_name + '\t' + bbox_info[:-1] + '\n')
        #     # car_txt.write('Object Detection2/' + img_name + '\t' + bbox_info[:-1] + '\n')
        # else:
        #     print('图像中没有渣土车:', img_path)
print('框的数量', number)








