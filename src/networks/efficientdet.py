from keras import layers
from keras import models
import os
import sys

sys.path.append(os.path.dirname(__file__))
from efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from efficientnet import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
from bifpn import build_wBiFPN
from head import regression_coco, classification_coco, regression_sand, classification_sand, properties_sand
from layers import ClipBoxes, RegressBoxes, FilterDetections
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from keras.layers import Conv2D
from keras.layers import Convolution2D
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]


def efficientdet(num_anchors, num_classes, num_properties, w_bifpn, d_bifpn, d_head, score_threshold, nms_threshold):
    image_input = layers.Input(shape=(None, None, 3))
    w_head = w_bifpn
    backbone_cls = backbones[0]
    # [(?, 256, 256, 16), (?, 128, 128, 24),(?, 64, 64, 24),(?, 32,32, 24),(?, 16, 16, 24)]
    features = backbone_cls(input_tensor=image_input)

    fpn_features = features
    for i in range(d_bifpn):
        fpn_features = build_wBiFPN(fpn_features, w_bifpn, i)
    reg = regression_coco(fpn_features, w_head, d_head, num_anchors)
    cls = classification_coco(fpn_features, w_head, d_head, num_anchors, num_classes)
    pro = properties_sand(fpn_features, w_head, d_head, num_anchors, num_properties)
    model = models.Model(inputs=[image_input], outputs=[reg, cls, pro], name='efficientdet')

    anchors_input = layers.Input((None, 4), name='anchors_input')
    boxes = RegressBoxes(name='boxes')([anchors_input, reg])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    # boxes (?, 49104, 4) (?, 49104, 1) (?, 49104, 3)
    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        class_specific_filter=True,
        max_detections=100
    )([boxes, cls, pro])

    prediction_model = models.Model(inputs=[image_input, anchors_input], outputs=detections, name='efficientdet_p')
    return model, prediction_model


def efficientdet_sand(num_anchors, num_classes, num_properties, w_bifpn, d_bifpn, d_head, score_threshold, nms_threshold):
    image_input = layers.Input(shape=(None, None, 3))
    w_head = w_bifpn
    backbone_cls = backbones[0]
    # [(?, 256, 256, 16), (?, 128, 128, 24),(?, 64, 64, 24),(?, 32,32, 24),(?, 16, 16, 24)]
    features = backbone_cls(input_tensor=image_input)

    fpn_features = features
    for i in range(d_bifpn):
        fpn_features = build_wBiFPN(fpn_features, w_bifpn, i)
    reg = regression_coco(fpn_features, w_head, d_head, num_anchors)
    cls = classification_coco(fpn_features, w_head, d_head, num_anchors, 90)
    coco_model = models.Model(inputs=[image_input], outputs=[reg, cls], name='efficientdet_coco')
    path = os.path.join(os.path.dirname(__file__), 'weights/efficientdet-d0.h5')
    coco_model.load_weights(path, by_name=True)
    # for i in range(1, 227):  # 321
    #     coco_model.layers[i].trainable = False
        # coco_model.layers[i].training = False

    P3_out = coco_model.get_layer(name='fpn_cells/cell_2/fnode3/op_after_combine8/bn').output
    P4_td = coco_model.get_layer(name='fpn_cells/cell_2/fnode2/op_after_combine7/bn').output
    P5_td = coco_model.get_layer(name='fpn_cells/cell_2/fnode1/op_after_combine6/bn').output
    P6_td = coco_model.get_layer(name='fpn_cells/cell_2/fnode0/op_after_combine5/bn').output
    P7_out = coco_model.get_layer(name='fpn_cells/cell_2/fnode7/op_after_combine12/bn').output

    tmp_fpn_features = [P3_out, P4_td, P5_td, P6_td, P7_out]
    sand_reg = regression_sand(tmp_fpn_features, w_head, d_head, num_anchors)
    sand_cls = classification_sand(tmp_fpn_features, w_head, d_head, num_anchors, num_classes)
    sand_pro = properties_sand(tmp_fpn_features, w_head, d_head, num_anchors, num_properties)
    sand_model = models.Model(inputs=[image_input], outputs=[sand_reg, sand_cls, sand_pro], name='efficientdet_sand')

    anchors_input = layers.Input((None, 4), name='anchors_input')
    boxes = RegressBoxes(name='boxes')([anchors_input, sand_reg])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    # boxes (?, 49104, 4) (?, 49104, 1) (?, 49104, 3)
    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        class_specific_filter=True,
        max_detections=100
    )([boxes, sand_cls, sand_pro])

    prediction_model = models.Model(inputs=[image_input, anchors_input], outputs=detections, name='efficientdet_p')

    return sand_model, prediction_model
