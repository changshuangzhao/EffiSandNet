import json
import os

# car_class = ['小轿车', '面包车', '货车', '小卡车', '大卡车', 'SUV', '客车', '公交车', '出租车', '垃圾车', '汽油罐车', '渣土车', '渣土车且小卡车', '工程车', '厢式卡车', '洒水车', '水泥罐车', '其他']
other_car = ['小轿车', '面包车', '货车', '小卡车', '大卡车', 'SUV', '客车', '公交车', '出租车', '垃圾车', '汽油罐车', '工程车', '厢式卡车', '洒水车', '水泥罐车', '其他']

root = '/Users/yanyan/data/object_detection'

car_txt = open('other_car.txt', 'w')
number = 0

video_images = os.path.join(root, 'images')
video_annos = os.path.join(root, 'annotations')
video_per_annos = os.listdir(video_annos)
for anno_file in video_per_annos:
    if '.DS_Store' in anno_file:
        continue
    anno_path = os.path.join(video_annos, anno_file)

    with open(anno_path, 'r', encoding='utf8') as f:
        json_data = json.load(f)
        # 当前图像的路径信息
        img_path = '/'.join(json_data['data']['image_url'].split('/')[-2:])

        img_name = img_path.split('/')[1]
        img_infos = json_data['result']['data']

        bbox_number = len(img_infos)
        for index, img_info in enumerate(img_infos):
            bbox_label = img_info['label']
            bbox_category = img_info['category']
            category2label = {}
            for category, label in zip(bbox_category, bbox_label):
                category2label[category] = label

            if '渣土车' in str(category2label.get('车辆标注-车辆类型', None)) and category2label.get('车辆标注-渣土车是否苫盖', None) is not None:
                break
            elif str(category2label.get('车辆标注-车辆类型', None)) in other_car:
                if index == (bbox_number - 1):
                    car_txt.write(f'object_detection/images/{img_name}' + '\n')








