import keras
import tensorflow as tf
import argparse
import os
import sys
from datetime import date
import keras.backend as K
from keras.optimizers import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '../generators'))
from generator import CSVGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from efficientdet import efficientdet_sand, efficientdet
from losses import bce, focal, smooth_l1
from callbacks import create_callbacks
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

def params():
    parser = argparse.ArgumentParser(description='EfficientDet for SandCar Training With Keras')
    parser.add_argument('--gpus', '-g', default='0', type=str,
                        help='GPU ID.')
    parser.add_argument('--epochs', '-e', default=10000, type=int,
                        help='Numbers of epochs to train.')
    parser.add_argument('--batch_size', '-b', default=1, type=int,
                        help='Batch size for training.')
    parser.add_argument('--snapshot', default='imagenet',
                        help='Resume training from a snapshot.')
    parser.add_argument('--score_threshold', '-s', default=0.01, type=float,
                        help='Threshold of box score.')
    parser.add_argument('--nms_threshold', '-n', default=0.5, type=float,
                        help='Threshold of nms iou.')
    parser.add_argument('--random_transform', default=False,
                        help='Randomly transform image and annotations.')
    return parser.parse_args()


def main():
    args = params()
    if args.gpus:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpus}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    if args.random_transform:
        print('random transform')
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        print('no random transform')
        misc_effect = None
        visual_effect = None

    train_generator = CSVGenerator(base_dir=cfg.DataRoot, data_file=cfg.TrainData, class_file=cfg.Cls, property_file=cfg.Pro,
                              batch_size=args.batch_size, image_sizes=cfg.InputSize_w, misc_effect=misc_effect, visual_effect=visual_effect)
    val_generator = CSVGenerator(base_dir=cfg.DataRoot, data_file=cfg.ValData, class_file=cfg.Cls, property_file=cfg.Pro,
                            batch_size=args.batch_size, image_sizes=cfg.InputSize_w, shuffle_groups=False)

    num_classes = train_generator.num_classes()
    num_properties = train_generator.num_properties()
    num_anchors = train_generator.num_anchors

    model, prediction_model = efficientdet(num_anchors, num_classes, num_properties, cfg.w_bifpn, cfg.d_bifpn, cfg.d_head, args.score_threshold, args.nms_threshold)
    # model, prediction_model = efficientdet(num_anchors, num_classes, num_properties, cfg.w_bifpn, cfg.d_bifpn, cfg.d_head, args.score_threshold, args.nms_threshold)
    # model.load_weights(os.path.join(os.getcwd(), '../networks/weights/efficientdet-d0.h5'), by_name=True, skip_mismatch=True)
    if args.snapshot == 'imagenet':
        coco_model_name = 'efficientnet-b0'
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(coco_model_name)
        file_hash = WEIGHTS_HASHES[coco_model_name][1]
        weights_path = keras.utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='coco_models', file_hash=file_hash)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    else:
        model.load_weights(args.snapshot, by_name=True, skip_mismatch=True)
        print('Loading model, this may take a second...')

    for i in range(1, 227):  # 321
        model.layers[i].trainable = False
        # model.layers[i].training = False

    model.summary()
    # for i in range(len(model.layers)):
    #     print(model.layers[i].name)

    if args.gpus and len(args.gpus.split(',')) > 1:
        model = keras.utils.multi_gpu_model(model, gpus=list(map(int, args.gpu.split(','))))

    model.compile(optimizer=Adam(lr=1e-4), loss={'regression': smooth_l1(), 'classification': focal(), 'pro': bce()})

    callbacks = create_callbacks(prediction_model, val_generator)
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=max(1, train_generator.size() // args.batch_size),
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
        validation_data=val_generator,
        validation_steps=max(1, val_generator.size() // args.batch_size)
    )


if __name__ == '__main__':
    main()


