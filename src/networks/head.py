import sys
import os
from keras import layers
from keras import models
import tensorflow as tf
from keras import initializers

sys.path.append(os.path.dirname(__file__))
from initializers import PriorProbability


MOMENTUM = 0.997
EPSILON = 1e-4


def regression_coco(fpn_features, w_head, d_head, num_anchors):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'depthwise_initializer': initializers.VarianceScaling(),
        'pointwise_initializer': initializers.VarianceScaling(),
    }
    box_convs = [layers.SeparableConv2D(filters=w_head, bias_initializer='zeros', name=f'box_net/box-{i}', **options) for i in range(d_head)]
    box_head_conv = layers.SeparableConv2D(filters=4 * num_anchors,
                                           bias_initializer='zeros',
                                           name=f'box_net/box-predict', **options)
    box_bns = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'box_net/box-{i}-bn-{j}') for j in range(3, 8)] for i in range(d_head)]

    box_relu = layers.Lambda(lambda x: tf.nn.swish(x))
    box_reshape = layers.Reshape((-1, 4))
    regression = []
    for i, feature in enumerate(fpn_features):
        for j in range(d_head):
            feature = box_convs[j](feature)
            feature = box_bns[j][i](feature)
            feature = box_relu(feature)
        feature = box_head_conv(feature)
        feature = box_reshape(feature)
        regression.append(feature)
    regression = layers.Concatenate(axis=1, name='regression')(regression)
    return regression


def classification_coco(fpn_features, w_head, d_head, num_anchors, num_classes):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'depthwise_initializer': initializers.VarianceScaling(),
        'pointwise_initializer': initializers.VarianceScaling(),
    }
    cls_convs = [layers.SeparableConv2D(filters=w_head, bias_initializer='zeros', name=f'class_net/class-{i}', **options) for i in range(d_head)]
    cls_head_conv = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                           bias_initializer=PriorProbability(probability=3e-4),
                                           name='class_net/class-predict', **options)
    cls_bns = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'class_net/class-{i}-bn-{j}') for j in range(3, 8)] for i in range(d_head)]
    cls_relu = layers.Lambda(lambda x: tf.nn.swish(x))
    classification = []
    cls_reshape = layers.Reshape((-1, num_classes))
    cls_activation = layers.Activation('sigmoid')
    for i, feature in enumerate(fpn_features):
        for j in range(d_head):
            feature = cls_convs[j](feature)
            feature = cls_bns[j][i](feature)
            feature = cls_relu(feature)
        feature = cls_head_conv(feature)
        feature = cls_reshape(feature)
        feature = cls_activation(feature)
        classification.append(feature)
    classification = layers.Concatenate(axis=1, name='classification')(classification)
    return classification


def regression_sand(fpn_features, w_head, d_head, num_anchors):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'depthwise_initializer': initializers.VarianceScaling(),
        'pointwise_initializer': initializers.VarianceScaling(),
    }
    box_convs = [layers.SeparableConv2D(filters=w_head, bias_initializer='zeros', name=f'box_net/box-{i}', **options) for i in range(d_head)]
    box_head_conv = layers.SeparableConv2D(filters=4 * num_anchors,
                                           bias_initializer='zeros',
                                           name=f'box_net/box-predict', **options)
    box_bns = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'box_net/box-{i}-bn-{j}') for j in range(3, 8)] for i in range(d_head)]

    box_relu = layers.Lambda(lambda x: tf.nn.swish(x))
    box_reshape = layers.Reshape((-1, 4))
    regression = []
    for i, feature in enumerate(fpn_features):
        for j in range(d_head):
            feature = box_convs[j](feature)
            feature = box_bns[j][i](feature)
            feature = box_relu(feature)
        feature = box_head_conv(feature)
        feature = box_reshape(feature)
        regression.append(feature)
    regression = layers.Concatenate(axis=1, name='regression_sand')(regression)
    return regression


def classification_sand(fpn_features, w_head, d_head, num_anchors, num_classes):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'depthwise_initializer': initializers.VarianceScaling(),
        'pointwise_initializer': initializers.VarianceScaling(),
    }
    cls_convs = [layers.SeparableConv2D(filters=w_head, bias_initializer='zeros', name=f'class_net/class-{i}', **options) for i in range(d_head)]
    cls_head_conv = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                           bias_initializer=PriorProbability(probability=3e-4),
                                           name='class_net/class-predict', **options)
    cls_bns = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'class_net/class-{i}-bn-{j}') for j in range(3, 8)] for i in range(d_head)]
    cls_relu = layers.Lambda(lambda x: tf.nn.swish(x))
    classification = []
    cls_reshape = layers.Reshape((-1, num_classes))
    cls_activation = layers.Activation('sigmoid')
    for i, feature in enumerate(fpn_features):
        for j in range(d_head):
            feature = cls_convs[j](feature)
            feature = cls_bns[j][i](feature)
            feature = cls_relu(feature)
        feature = cls_head_conv(feature)
        feature = cls_reshape(feature)
        feature = cls_activation(feature)
        classification.append(feature)
    classification = layers.Concatenate(axis=1, name='classification_sand')(classification)
    return classification


def properties_sand(fpn_features, w_head, d_head, num_anchors, num_properties):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'depthwise_initializer': initializers.VarianceScaling(),
        'pointwise_initializer': initializers.VarianceScaling(),
    }
    pro_convs = [
        layers.SeparableConv2D(filters=w_head, bias_initializer='zeros', name=f'property_net/property-{i}', **options)
        for i in range(d_head)]
    pro_head_conv = layers.SeparableConv2D(filters=num_properties * num_anchors,
                                           bias_initializer='zeros',
                                           name='property_net/property-predict', **options)
    pro_bns = [
        [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'property_net/property-{i}-bn-{j}') for j
         in range(3, 8)] for i in range(d_head)]
    pro_relu = layers.Lambda(lambda x: tf.nn.swish(x))
    pro = []
    pro_reshape = layers.Reshape((-1, num_properties))
    pro_activation = layers.Activation('softmax')
    for i, feature in enumerate(fpn_features):
        for j in range(d_head):
            feature = pro_convs[j](feature)
            feature = pro_bns[j][i](feature)
            feature = pro_relu(feature)
        feature = pro_head_conv(feature)
        feature = pro_reshape(feature)
        feature = pro_activation(feature)
        pro.append(feature)
    pro = layers.Concatenate(axis=1, name='pro_sand')(pro)
    return pro