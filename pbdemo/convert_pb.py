import os
import sys
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/networks'))
from efficientdet import efficientdet_sand


def save_as_pb(model):
    K.set_learning_phase(0)
    orig_output_node_names = [node.op.name for node in model.outputs]

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        orig_output_node_names)

    graph_io.write_graph(constant_graph, 'pbfile', 'sand.pb', as_text=False)


def main():
    weight_file_path = '../src/train/checkpoints/2020-08-30/csv_212_1.1373_1.5797.h5'
    sand_model, _ = efficientdet_sand(num_anchors=9, num_classes=1, num_properties=3, w_bifpn=64, d_bifpn=3, d_head=3, score_threshold=0.01, nms_threshold=0.5)
    sand_model.load_weights(weight_file_path, by_name=True)
    save_as_pb(sand_model)
    print('model saved')


if __name__ == '__main__':
    main()



