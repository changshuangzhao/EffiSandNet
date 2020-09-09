"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cv2
import numpy as np

identity_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def colvec(*args):
    """
    Create a numpy array representing a column vector.
    """
    return np.array([args]).T


def transform_aabb(transform_matrix, aabb):
    """
    Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    Args
        transform: The transformation to apply.
        x1: The minimum x value of the AABB.
        y1: The minimum y value of the AABB.
        x2: The maximum x value of the AABB.
        y2: The maximum y value of the AABB.
    Returns
        The new AABB as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform_matrix.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1, 1, 1, 1],
    ])

    # Extract the min and max corners again.
    # (3, ) (min_x, min_y, 1)
    min_corner = points.min(axis=1)
    # (3, ) (max_x, max_y, 1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def random_value(min, max):
    return np.random.uniform(min, max)


def random_vector(min, max):
    """
    Construct a random vector between min and max.

    Args
        min: the minimum value for each component, (n, )
        max: the maximum value for each component, (n, )
    """
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return np.random.uniform(min, max)


def rotation(min=0, max=0, prob=0.5):
    """
    Construct a homogeneous 2D rotation matrix.

    Args
        min: a scalar for the minimum absolute angle in radians
        max: a scalar for the maximum absolute angle in radians
    Returns
        the rotation matrix as 3 by 3 numpy array
    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # angle: the angle in radians
        angle = random_value(min=min, max=max)
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def translation_x(min=0, max=0, prob=0.5):
    """
    Construct a homogeneous 2D translation matrix.

    Args:
        min: a scalar for the minimum translation for x axis
        max: a scalar for the maximum translation for x axis

    Returns:
        the translation matrix as 3 by 3 numpy array

    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # translation: the translation 2D vector
        translation = random_value(min=min, max=max)
        return np.array([
            [1, 0, translation],
            [0, 1, ],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def translation_y(min=0, max=0, prob=0.5):
    """
    Construct a homogeneous 2D translation matrix.

    Args:
        min: a scalar for the minimum translation for y axis
        max: a scalar for the maximum translation for y axis

    Returns:
        the translation matrix as 3 by 3 numpy array

    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # translation: the translation 2D vector
        translation = random_value(min=min, max=max)
        return np.array([
            [1, 0],
            [0, 1, translation],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def translation_xy(min=(0, 0), max=(0, 0), prob=0.5):
    """
    Construct a homogeneous 2D translation matrix.

    Args:
        min: a scalar for the minimum translation for y axis
        max: a scalar for the maximum translation for y axis

    Returns:
        the translation matrix as 3 by 3 numpy array

    """
    random_prob = np.random.uniform()
    if random_prob < prob:
        # translation: the translation 2D vector
        dx = np.random.randint(min[0], max[0])
        dy = np.random.randint(min[1], max[1])
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def shear_x(min=0, max=0, prob=0.5):
    """
    Construct a homogeneous 2D shear matrix.

    Args
        min:  the minimum shear angle in radians.
        max:  the maximum shear angle in radians.
    Returns
        the shear matrix as 3 by 3 numpy array
    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # angle: the shear angle in radians
        angle = random_value(min=min, max=max)
        return np.array([
            [1, np.tan(angle), 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def shear_y(min, max, prob=0.5):
    """
    Construct a homogeneous 2D shear matrix.

    Args
        min:  the minimum shear angle in radians.
        max:  the maximum shear angle in radians.
    Returns
        the shear matrix as 3 by 3 numpy array
    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # angle: the shear angle in radians
        angle = random_value(min=min, max=max)
        return np.array([
            [1, 0, 0],
            [np.tan(angle), 1, 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def scaling_x(min=0.9, max=1.1, prob=0.5):
    """
    Construct a homogeneous 2D scaling matrix.

    Args
        factor: a 2D vector for X and Y scaling
    Returns
        the zoom matrix as 3 by 3 numpy array
    """

    random_prob = np.random.uniform()
    if random_prob > prob:
        # angle: the shear angle in radians
        factor = random_value(min=min, max=max)
        return np.array([
            [factor, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def scaling_y(min=0.9, max=1.1, prob=0.5):
    """
    Construct a homogeneous 2D scaling matrix.

    Args
        factor: a 2D vector for X and Y scaling
    Returns
        the zoom matrix as 3 by 3 numpy array
    """

    random_prob = np.random.uniform()
    if random_prob > prob:
        # angle: the shear angle in radians
        factor = random_value(min=min, max=max)
        return np.array([
            [1, 0, 0],
            [0, factor, 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def scaling_xy(min=(0.9, 0.9), max=(1.1, 1.1), prob=0.5):
    """
    Construct a homogeneous 2D scaling matrix.

    Args
        min: a 2D vector containing the minimum scaling factor for X and Y.
        min: a 2D vector containing The maximum scaling factor for X and Y.
    Returns
        the zoom matrix as 3 by 3 numpy array
    """

    random_prob = np.random.uniform()
    if random_prob > prob:
        # factor: a 2D vector for X and Y scaling
        factor = random_vector(min=min, max=max)
        return np.array([
            [factor[0], 0, 0],
            [0, factor[1], 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def flip_x(prob=0.8):
    """
    Construct a transformation randomly containing X/Y flips (or not).

    Args
        flip_x_chance: The chance that the result will contain a flip along the X axis.
        flip_y_chance: The chance that the result will contain a flip along the Y axis.
    Returns
        a homogeneous 3 by 3 transformation matrix
    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # 1 - 2 * bool gives 1 for False and -1 for True.
        return np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def flip_y(prob=0.8):
    """
    Construct a transformation randomly containing X/Y flips (or not).

    Args
        flip_x_chance: The chance that the result will contain a flip along the X axis.
        flip_y_chance: The chance that the result will contain a flip along the Y axis.
    Returns
        a homogeneous 3 by 3 transformation matrix
    """
    random_prob = np.random.uniform()
    if random_prob > prob:
        # 1 - 2 * bool gives 1 for False and -1 for True.
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    else:
        return identity_matrix


def change_transform_origin(transform, center):
    """
    Create a new transform representing the same transformation, only with the origin of the linear part changed.

    Args
        transform: the transformation matrix
        center: the new origin of the transformation
    Returns
        translate(center) * transform * translate(-center)
    """
    center = np.array(center)
    return np.linalg.multi_dot([np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]]),
                                transform,
                                np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])])


def random_transform(
        min_rotation=0,
        max_rotation=0,
        min_translation=(0, 0),
        max_translation=(0, 0),
        min_shear=0,
        max_shear=0,
        min_scaling=(1, 1),
        max_scaling=(1, 1),
):
    """
    Create a random transformation.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
     as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation:    The minimum rotation in radians for the transform as scalar.
        max_rotation:    The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear angle for the transform in radians.
        max_shear:       The maximum shear angle for the transform in radians.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
    """
    return np.linalg.multi_dot([
        rotation(min_rotation, max_rotation),
        translation_xy(min_translation, max_translation),
        shear_x(min_shear, max_shear) if np.random.uniform() > 0.5 else shear_y(min_shear, max_shear),
        scaling_xy(min_scaling, max_scaling),
        flip_x() if np.random.uniform() > 0.5 else flip_y(),
    ])


def random_transform_generator(**kwargs):
    """
    Create a random transform generator.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation: The minimum rotation in radians for the transform as scalar.
        max_rotation: The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear: The minimum shear angle for the transform in radians.
        max_shear: The maximum shear angle for the transform in radians.
        min_scaling: The minimum scaling for the transform as 2D column vector.
        max_scaling: The maximum scaling for the transform as 2D column vector.
    """

    while True:
        yield random_transform(**kwargs)


def adjust_transform_for_image(transform, image, relative_translation):
    """
    Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """
    Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode: One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation: One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval: Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """

    def __init__(
            self,
            fill_mode='nearest',
            interpolation='linear',
            cval=0,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cv_border_mode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cv_interpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )
    return output


class RandomCrop():
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
    """

    def __init__(self,
                 aspect_ratio=(.5, 2.),
                 thresholds=(.0, .1, .3, .5, .7, .9),
                 scaling=(.3, 1.),
                 num_attempts=30,
                 allow_no_crop=True,
                 cover_all_box=False,
                 ):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, image, boxes, classes, crop_h_=None, crop_w_=None):
        if len(boxes) == 0:
            return image, boxes, classes


        h, w, _ = np.shape(image)
        gt_bbox = boxes

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return image, boxes, classes

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                min_ar, max_ar = self.aspect_ratio
                aspect_ratio = np.random.uniform(
                    max(min_ar, scale**2), min(max_ar, scale**-2))
                if crop_h_ is not None:
                    crop_h = min(crop_h_, h)
                else:
                    crop_h = int(h * scale / np.sqrt(aspect_ratio))

                if crop_w_ is not None:
                    crop_w = min(crop_w_, w)
                else:
                    crop_w = int(w * scale * np.sqrt(aspect_ratio))

                if h > crop_h:
                    crop_y = np.random.randint(0, h - crop_h)
                else:
                    crop_y = 0
                if w > crop_w:
                    crop_x = np.random.randint(0, w - crop_w)
                else:
                    crop_x = 0
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break
            if found:
                image = self._crop_image(image, crop_box)
                boxes = np.take(cropped_box, valid_ids, axis=0)
                classes = np.take(
                    classes, valid_ids, axis=0)
                #sample['w'] = crop_box[2] - crop_box[0]
                #sample['h'] = crop_box[3] - crop_box[1]
                return image, boxes, classes

        return image, boxes, classes

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]
