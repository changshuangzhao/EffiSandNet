import argparse
import json
import cv2
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))
import coco_tabel as table

COCODataNames = ['truck']


def main(args):
    out_file = args.out_file
    file_w = open(out_file, 'w')
    record = open(args.recordfile, 'w')
    with open(args.annotation_path) as f:
        data = json.load(f)
    annotations = data['annotations']
    total = len(annotations)
    image_name_dic = defaultdict(list)
    instance_cnt_dic = defaultdict(lambda: 0)
    tmp_dir = args.image_dir.strip().split('/')[-1]
    for t_id, annotation in enumerate(annotations):
        sys.stdout.write("\r>>>> process %d / %d" % (t_id, total))
        sys.stdout.flush()
        catid = annotation['category_id']
        # clsid = table.mscoco2017[catid][0]
        clsname = table.mscoco2017[catid][1]
        cls_split = clsname.strip().split()
        if len(cls_split) > 0:
            clsname = '_'.join(cls_split)
        if clsname in COCODataNames:
            clsid = COCODataNames.index(clsname)
        else:
            continue
        image_filename = '{0:012d}'.format(annotation['image_id']) + '.jpg'

        src = os.path.join(args.image_dir, image_filename)
        if not os.path.exists(src):
            print('not exist : ', src)
            continue
        # img = cv2.imread(src)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # h, w = img.shape[:2]
        bbox = annotation['bbox']
        '''
        x1 = bbox[0] / w
        y1 = bbox[1] / h
        x2 = (bbox[0] + bbox[2]) / w
        y2 = (bbox[1] + bbox[3]) / h
        '''
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        # label = [str(clsid), str(x1), str(y1), str(x2), str(y2)]
        image_name_dic[image_filename].extend([x1, y1, x2, y2, clsid])
        instance_cnt_dic[clsname] += 1
        # output_filename = os.path.splitext(image_filename)[0] + '.txt'
        # dst = os.path.join(out_dir, output_filename)
        # with open(dst, 'a') as f:
        #   f.write('\t'.join(label) + '\n')
        # print(label, src)
    for tmp_key in image_name_dic.keys():
        cls_bb = map(str, image_name_dic[tmp_key])
        cls_bb_str = ','.join(cls_bb)
        imgpath = os.path.join(tmp_dir, tmp_key)
        # imgpath = tmp_key
        file_w.write("{},{}\n".format(imgpath, cls_bb_str))
    for tmp_key in sorted(instance_cnt_dic.keys()):
        record.write('{},{}\n'.format(tmp_key, str(instance_cnt_dic[tmp_key])))
    record.write('total_imgs,{}'.format(len(image_name_dic.keys())))
    file_w.close()
    f.close()
    record.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='/Users/yanyan/data/COCO2017/images/train2017')
    parser.add_argument('--annotation_path', default='/Users/yanyan/data/COCO2017/annotations/instances_train2017.json')
    parser.add_argument('--out_file', default='out_file.txt')
    parser.add_argument('--recordfile', default='recordfile.txt')
    main(parser.parse_args())
