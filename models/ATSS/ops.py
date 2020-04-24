import mxnet as mx
import math


def _corner_to_center(F, boxes):
    boxes = F.slice_axis(boxes, axis=-1, begin=-4, end=None)
    xmin, ymin, xmax, ymax = F.split(boxes, axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) * 0.5
    y = ymin + (height - 1) * 0.5
    return x, y, width, height


def _decode_bbox(F, anchors, deltas, means, stds, class_agnostic):
    """
    anchors: (#img, #roi, #cls * 4)
    deltas: (#img, #roi, #cls * 4)
    im_infos: (#img, 3), [h, w, scale]
    means: (4, ), [x, y, h, w]
    stds: (4, ), [x, y, h, w]
    class_agnostic: bool
    Returns:
    bbox: (#img, #roi, 4), [x1, y1, x2, y2]
    """
    with mx.name.Prefix("decode_bbox: "):
        if class_agnostic:
            # TODO: class_agnostic should predict only 1 class
            # class_agnostic predicts 2 classes
            deltas = F.slice_axis(deltas, axis=-1, begin=-4, end=None)
        if not class_agnostic:
            # add class axis, layout (img, roi, cls, coord)
            deltas = F.reshape(deltas, [0, 0, -4, -1, 4])  # TODO: extend to multiple anchors
            anchors = F.expand_dims(anchors, axis=-2)

        ax, ay, aw, ah = _corner_to_center(F, anchors)  # anchor
        dx, dy, dw, dh = F.split(deltas, axis=-1, num_outputs=4)

        # delta
        dx = dx * stds[0] + means[0]
        dy = dy * stds[1] + means[1]
        dw = dw * stds[2] + means[2]
        dh = dh * stds[3] + means[3]

        # prevent large numbers
        max_ratio = math.log(1000. / 16)
        dw = F.clip(dw, -max_ratio, max_ratio)
        dh = F.clip(dh, -max_ratio, max_ratio)

        # prediction
        px = F.broadcast_add(F.broadcast_mul(dx, aw), ax)
        py = F.broadcast_add(F.broadcast_mul(dy, ah), ay)
        pw = F.broadcast_mul(F.exp(dw), aw)
        ph = F.broadcast_mul(F.exp(dh), ah)

        x1 = px - 0.5 * (pw - 1.0)
        y1 = py - 0.5 * (ph - 1.0)
        x2 = px + 0.5 * (pw - 1.0)
        y2 = py + 0.5 * (ph - 1.0)

        out = F.concat(x1, y1, x2, y2, dim=-1)
        if not class_agnostic:
            out = F.reshape(out, [0, 0, -3, -2])
    return out


def _giou_loss(F, pred, target, eps=1e-3):
    """ giou loss
    inputs:
        F: symbol or ndarray
        pred: F, (#img, ..., 4)
        target: F, (#img, ..., 4)
        eps: float
    outputs:
        loss: F, (#img, ...)
    """
    p_x1, p_y1, p_x2, p_y2 = F.split(pred, num_outputs=4, axis=-1)
    t_x1, t_y1, t_x2, t_y2 = F.split(target, num_outputs=4, axis=-1)

    # inter
    inter_w = F.maximum(F.minimum(p_x2, t_x2) - F.maximum(p_x1, t_x1) + 1., 0.)
    inter_h = F.maximum(F.minimum(p_y2, t_y2) - F.maximum(p_y1, t_y1) + 1., 0.)
    inter_area = inter_w * inter_h

    # union
    p_area = (p_x2 - p_x1 + 1.) * (p_y2 - p_y1 + 1.)
    t_area = (t_x2 - t_x1 + 1.) * (t_y2 - t_y1 + 1.)
    union_area = p_area + t_area - inter_area + eps

    # iou
    ious = inter_area / union_area

    # enclose
    enclose_w = F.maximum(F.maximum(p_x2, t_x2) - F.minimum(p_x1, t_x1) + 1., 0.)
    enclose_h = F.maximum(F.maximum(p_y2, t_y2) - F.minimum(p_y1, t_y1) + 1., 0.)
    enclose_area = enclose_w * enclose_h + eps

    # giou
    gious = ious - (enclose_area - union_area) / enclose_area  # range in [-1, 1]

    return F.clip(1. - gious, 0, 1)


if __name__ == "__main__":
    F = mx.nd
    pred = F.array([[0, 0, 100, 100], [50, 80, 60, 100], [-1, -1, -1, -1], [0, 0, 0, 0]])
    target = F.array([[0, 10, 20, 30], [40, 50, 60, 70], [-2, -2, -2, -2], [0, 0, 0, 0]])
    loss = _giou_loss(F, pred, target)
    print(loss)
    means = [0, 0, 0, 0]
    stds = [1, 1, 1, 1]
    np_pred = pred.asnumpy()
    np_target = target.asnumpy()
    from operator_py.bbox_transform import nonlinear_transform
    np_delta = nonlinear_transform(np_pred, np_target)
    print(np_delta)
    delta = F.array(np_delta)
    output = _decode_bbox(F, pred, delta, means, stds, True)
    print(output)
