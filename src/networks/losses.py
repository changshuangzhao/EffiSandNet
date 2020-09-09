import keras
import tensorflow as tf
from keras import backend as K

def bce():
    def _bce(y_true, y_pred):
        pro_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        pro = y_pred

        indices = tf.where(keras.backend.equal(anchor_state, 1))
        pro = tf.gather_nd(pro, indices)
        pro_target = tf.gather_nd(pro_target, indices)
        # pro_loss = keras.losses.categorical_crossentropy(pro_target, pro, from_logits=True)
        # pro_loss = keras.losses.binary_crossentropy(pro_target, pro, from_logits=True)
        pro_loss = keras.backend.categorical_crossentropy(pro_target, pro, from_logits=False)
        # pro_loss = keras.backend.binary_crossentropy(pro_target, pro, from_logits=True)
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(pro_loss) / normalizer
    return _bce


def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def diou():
    def _diou(y_true, y_pred):

        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        dious = torch.zeros((rows, cols))
        if rows * cols == 0:  #
            return dious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            dious = torch.zeros((cols, rows))
            exchange = True
        # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
        w1 = bboxes1[:, 2] - bboxes1[:, 0]
        h1 = bboxes1[:, 3] - bboxes1[:, 1]
        w2 = bboxes2[:, 2] - bboxes2[:, 0]
        h2 = bboxes2[:, 3] - bboxes2[:, 1]

        area1 = w1 * h1
        area2 = w2 * h2

        center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
        center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
        center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
        center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

        inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
        union = area1 + area2 - inter_area
        dious = inter_area / union - (inter_diag) / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)
        if exchange:
            dious = dious.T
        return dious
    return _diou