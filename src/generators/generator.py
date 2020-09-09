import cv2
import numpy as np
from PIL import Image
from six import raise_from
import csv
import os.path as osp
from collections import OrderedDict
import os
import sys
sys.path.append(os.path.dirname(__file__))
from common import Generator


class CSVGenerator(Generator):
    """
    Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
            self,
            base_dir,
            data_file,
            class_file,
            property_file,
            **kwargs
    ):
        """
        Initialize a CSV data generator.

        Args
            data_file: Path to the CSV annotations file.
            class_file: Path to the CSV classes file.
            detect_text: if do text detection
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the data_file).
        """

        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            if osp.exists(data_file):
                self.base_dir = ''
            else:
                self.base_dir = osp.dirname(data_file)

        # parse the provided class file
        try:
            with _open_for_csv(class_file) as file:
                # class_name --> class_id
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(class_file, e)), None)

        self.labels = {}
        # class_id --> class_name
        for key, value in self.classes.items():
            self.labels[value] = key

        # parse the provided property file
        try:
            with _open_for_csv(property_file) as file:
                # property_name --> property_id
                self.properties = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV property file: {}: {}'.format(property_file, e)), None)

        self.id2property = {}
        # class_id --> class_name
        for key, value in self.properties.items():
            self.id2property[value] = key

        # csv with img_path, x1, y1, x2, y2, x3, y3, x4, y4, class_name
        try:
            with _open_for_csv(data_file) as file:
                # {'img_path1':[{'x1':xx,'y1':xx,'x2':xx,'y2':xx,'x3':xx,'y3':xx,'x4':xx,'y4':xx, 'class':xx}...],...}
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes, self.properties)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(data_file, e)), None)
        self.image_names = list(self.image_data.keys())
        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def num_properties(self):
        """
        Number of properties in the dataset.
        """
        return max(self.properties.values()) + 1

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def name_to_property_id(self, name):
        """
        Map name to property.
        """
        return self.properties[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """
        Returns the image path for image_index.
        """
        return osp.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        image = cv2.imread(self.image_path(image_index))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0, ), dtype=np.int32),
                       'properties': np.empty((0, ), dtype=np.int32),
                       'bboxes': np.empty((0, 4), dtype=np.float32),
                       'quadrangles': np.empty((0, 4, 2), dtype=np.float32),
                       }
        # [('SandCar/images/cug_l_v0_f3400.jpg': [{'x1': 1150, 'y1': 234, 'x2': 1434, 'y2': 505, 'class': 'sand', 'property': 'no'}]),
        # ……,
        # ('SandCar/images/cug_l_v0_f3400.jpg': [{'x1': 1150, 'y1': 234, 'x2': 1434, 'y2': 505, 'class': 'sand', 'property': 'no'}])]

        # {'img_path1':[{'x1':xx,'y1':xx,'x2':xx,'y2':xx,'x3':xx,'y3':xx,'x4':xx,'y4':xx, 'class':xx}...],...}
        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['properties'] = np.concatenate((annotations['properties'], [self.name_to_property_id(annot['property'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))
        return annotations


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb', for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def _read_classes(csv_reader):
    """
    Parse the classes file given by csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_annotations(csv_reader, classes, properties):
    """
    Read annotations from the csv_reader.
    Args:
        csv_reader: csv reader of args.annotations_path
        classes: list[str] all the class names read from args.classes_path

    Returns:
        result: dict, dict is like {image_path: [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name}]}

    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader, 1):
        try:
            img_file, x1, y1, x2, y2, class_name, property_name = row[:10]
            if img_file not in result:
                result[img_file] = []
            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name, property_name) == ('', '', '', '', '', ''):
                continue

            x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if class_name not in classes:
                raise ValueError(f'line {line}: unknown class name: \'{class_name}\' (classes: {classes})')

            if property_name not in properties:
                raise ValueError(f'line {line}: unknown property name: \'{property_name}\' (properties: {properties})')

            result[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name, 'property': property_name})
        except ValueError:
            raise_from(ValueError(
                f'line {line}: format should be \'img_file,x1,y1,x2,y2,class_name,property\' or \'img_file,,,,,\''),
                None)

    return result


