from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from operator_py.cython.bbox import bbox_overlaps_cython
from operator_py.bbox_transform import nonlinear_transform as bbox_transform
from core.detection_input import AnchorTarget2D


class PyramidAnchorTarget2D(AnchorTarget2D):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 4)
    output: anchor_label, ndarray(num_anchor * h * w)
            anchor_bbox_target, ndarray(num_anchor * 4, h * w)
            anchor_bbox_weight, ndarray(num_anchor * 4, h * w)
    """

    def __init__(self, pAnchor):
        super().__init__(pAnchor)

        self.__base_anchor = None
        self.__v_all_anchor = None
        self.__h_all_anchor = None
        self.__num_anchor = None

        self.pyramid_levels = len(self.p.generate.stride)
        self.p_list = [copy.deepcopy(self.p) for _ in range(self.pyramid_levels)]

        pyramid_stride = self.p.generate.stride
        pyramid_short = self.p.generate.short
        pyramid_long = self.p.generate.long

        for i in range(self.pyramid_levels):
            self.p_list[i].generate.stride = pyramid_stride[i]
            self.p_list[i].generate.short = pyramid_short[i]
            self.p_list[i].generate.long = pyramid_long[i]

        self.anchor_target_2d_list = [AnchorTarget2D(p) for p in self.p_list]
        self.anchor_shape_list = [x.h_all_anchor.shape[0] for x in self.anchor_target_2d_list]

    def _gather_valid_anchor(self, image_info):
        valid_index_list = []
        valid_anchor_list = []
        for anchor_target_2d in self.anchor_target_2d_list:
            valid_index, valid_anchor = anchor_target_2d._gather_valid_anchor(image_info)
            valid_index_list.append(valid_index)
            valid_anchor_list.append(valid_anchor)

        return valid_index_list, valid_anchor_list

    def _assign_label_to_anchor(self, valid_anchor_list, gt_bbox, top_n):
        valid_anchor = np.concatenate(valid_anchor_list)
        valid_anchor_shape_list = [x.shape[0] for x in valid_anchor_list]
        num_anchor = valid_anchor.shape[0]
        cls_label = np.zeros(shape=(num_anchor,), dtype=np.float32)

        if len(gt_bbox) > 0 and num_anchor > 0:
            # compute iou between valid anchors and gt bboxes
            overlaps = bbox_overlaps_cython(valid_anchor.astype(
                np.float32, copy=False), gt_bbox.astype(np.float32, copy=False))  # (#A, #gt)

            # compute center distance between valid anchors and gt bboxes
            gt_cx = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2.0
            gt_cy = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2.0
            gt_centers = np.stack((gt_cx, gt_cy), axis=1)
            anchor_cx = (valid_anchor[:, 0] + valid_anchor[:, 2]) / 2.0
            anchor_cy = (valid_anchor[:, 1] + valid_anchor[:, 3]) / 2.0
            anchor_centers = np.stack((anchor_cx, anchor_cy), axis=1)
            distances = anchor_centers[:, None, :] - gt_centers[None, :, :]
            distances = np.sqrt(np.power(distances, 2).sum(-1))  # (#A, #gt)

            # select topk anchors for each gt bbox in each stride based on center distance
            candidate_index = []
            start_idx = 0
            for i in range(self.pyramid_levels):
                end_index = start_idx + valid_anchor_shape_list[i]
                distances_this_level = distances[start_idx: end_index, :]
                top_n_this_level = min(top_n, valid_anchor_shape_list[i])
                index_this_level = np.argpartition(
                    distances_this_level, kth=top_n_this_level - 1, axis=0)[:top_n_this_level]
                candidate_index.append(index_this_level + start_idx)
                start_idx = end_index
            candidate_index = np.concatenate(candidate_index, axis=0)  # (#levels * top_n, #gt)
            assert end_index == distances.shape[0]

            candidate_overlaps = overlaps[candidate_index, np.arange(len(gt_bbox))]
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0, ddof=1)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

            # select those samples whose center inside the gt
            candidate_centers = anchor_centers[candidate_index]
            l_inside = (candidate_centers[:, :, 0] - gt_bbox[None, :, 0]) > 0.01
            t_inside = (candidate_centers[:, :, 1] - gt_bbox[None, :, 1]) > 0.01
            r_inside = (gt_bbox[None, :, 2] - candidate_centers[:, :, 0]) > 0.01
            b_inside = (gt_bbox[None, :, 3] - candidate_centers[:, :, 1]) > 0.01
            is_inside = l_inside & t_inside & r_inside & b_inside
            is_pos = is_pos & is_inside

            # select the gt with highest iou with the positive anchors
            overlaps_pos = - np.ones_like(overlaps)
            candidate_overlaps = np.where(is_pos, candidate_overlaps, -1)
            overlaps_pos[candidate_index, np.arange(len(gt_bbox))] = candidate_overlaps
            max_overlaps = overlaps_pos.max(axis=1)
            argmax_overlaps = overlaps_pos.argmax(axis=1)
            # select positive samples
            cls_label[max_overlaps > -1] = gt_bbox[argmax_overlaps[max_overlaps > -1], 4]
        else:
            argmax_overlaps = np.zeros(shape=(num_anchor,))

        reg_target = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        centerness = np.zeros(shape=(num_anchor, 1), dtype=np.float32)
        box_weight = np.zeros(shape=(num_anchor, 1), dtype=np.float32)
        fg_index = np.where(cls_label > 0)[0]
        if len(fg_index) > 0:
            fg_anchor = valid_anchor[fg_index]
            fg_gt = gt_bbox[argmax_overlaps[fg_index], :4]
            reg_target[fg_index] = bbox_transform(
                fg_anchor, fg_gt, means=self.p.assign.mean, stds=self.p.assign.std)
            anchor_cx = (fg_anchor[:, 0] + fg_anchor[:, 2]) / 2.0
            anchor_cy = (fg_anchor[:, 1] + fg_anchor[:, 3]) / 2.0
            l_ = anchor_cx - fg_gt[:, 0]
            t_ = anchor_cy - fg_gt[:, 1]
            r_ = fg_gt[:, 2] - anchor_cx
            b_ = fg_gt[:, 3] - anchor_cy
            left_right = np.stack((l_, r_), axis=1)
            top_bottom = np.stack((t_, b_), axis=1)
            fg_centerness = np.sqrt(
                (np.min(left_right, axis=-1) / np.max(left_right, axis=-1)) *
                (np.min(top_bottom, axis=-1) / np.max(top_bottom, axis=-1))
            )
            centerness[fg_index, 0] = fg_centerness
            box_weight[fg_index, :] = 1.0

        cls_label_list = []
        reg_target_list = []
        centerness_list = []
        box_weight_list = []
        start_idx = 0
        for i in range(self.pyramid_levels):
            end_index = start_idx + valid_anchor_shape_list[i]
            cls_label_list.append(cls_label[start_idx: end_index])
            reg_target_list.append(reg_target[start_idx: end_index])
            centerness_list.append(centerness[start_idx: end_index])
            box_weight_list.append(box_weight[start_idx: end_index])
            start_idx = end_index

        return cls_label_list, reg_target_list, centerness_list, box_weight_list

    def _scatter_valid_anchor(self, valid_index_list, cls_label_list, reg_target_list,
                              centerness_list, box_weight_list, im_info):
        all_anchor_list = []
        all_cls_label_list = []
        all_reg_target_list = []
        all_centerness_list = []
        all_box_weight_list = []
        for i in range(self.pyramid_levels):
            num_anchor = self.anchor_shape_list[i]
            valid_index = valid_index_list[i]
            cls_label = cls_label_list[i]
            reg_target = reg_target_list[i]
            centerness = centerness_list[i]
            box_weight = box_weight_list[i]

            cls_label_level = np.full(shape=(num_anchor,), fill_value=-1, dtype=np.float32)
            reg_target_level = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
            centerness_level = np.zeros(shape=(num_anchor, 1), dtype=np.float32)
            box_weight_level = np.zeros(shape=(num_anchor, 1), dtype=np.float32)

            cls_label_level[valid_index] = cls_label
            reg_target_level[valid_index] = reg_target
            centerness_level[valid_index] = centerness
            box_weight_level[valid_index] = box_weight

            p = self.anchor_target_2d_list[i].p
            h, w = im_info[:2]
            if h >= w:
                fh, fw = p.generate.long, p.generate.short
                anchor_level = self.anchor_target_2d_list[i].v_all_anchor
            else:
                fh, fw = p.generate.short, p.generate.long
                anchor_level = self.anchor_target_2d_list[i].h_all_anchor
            anchor_level = anchor_level.reshape((fh, fw, -1)).transpose(2, 0, 1)
            cls_label_level = cls_label_level.reshape((fh, fw, -1)).transpose(2, 0, 1).reshape(-1)
            reg_target_level = reg_target_level.reshape((fh, fw, -1)).transpose(2, 0, 1)
            centerness_level = centerness_level.reshape((fh, fw, -1)).transpose(2, 0, 1)
            box_weight_level = box_weight_level.reshape((fh, fw, -1)).transpose(2, 0, 1)

            anchor_level = anchor_level.reshape(-1, fh * fw)
            reg_target_level = reg_target_level.reshape(-1, fh * fw)
            centerness_level = centerness_level.reshape(-1, fh * fw)
            box_weight_level = box_weight_level.reshape(-1, fh * fw)

            all_anchor_list.append(anchor_level)
            all_cls_label_list.append(cls_label_level)
            all_reg_target_list.append(reg_target_level)
            all_centerness_list.append(centerness_level)
            all_box_weight_list.append(box_weight_level)

        return all_anchor_list, all_cls_label_list, all_reg_target_list, \
            all_centerness_list, all_box_weight_list

    def apply(self, input_record):
        p = self.p

        im_info = input_record["im_info"]
        gt_bbox = input_record["gt_bbox"]
        h, w = im_info[:2]
        assert isinstance(gt_bbox, np.ndarray)
        assert gt_bbox.dtype == np.float32

        # select valid gt bboxes
        valid = np.where(gt_bbox[:, 0] != -1)[0]
        gt_bbox = gt_bbox[valid]

        # select valid anchors
        valid_index_list, valid_anchor_list = self._gather_valid_anchor(im_info)

        # adaptively select training sample
        cls_label_list, reg_target_list, centerness_list, box_weight_list = \
            self._assign_label_to_anchor(valid_anchor_list, gt_bbox, p.assign.top_n)

        # scatter
        anchor_list, cls_label_list, reg_target_list, centerness_list, box_weight_list = \
            self._scatter_valid_anchor(
                valid_index_list, cls_label_list, reg_target_list, centerness_list,
                box_weight_list, im_info)

        anchor = np.concatenate(anchor_list, axis=1)
        cls_label = np.concatenate(cls_label_list, axis=0)
        reg_target = np.concatenate(reg_target_list, axis=1)
        centerness = np.concatenate(centerness_list, axis=1)
        box_weight = np.concatenate(box_weight_list, axis=1)

        input_record["rpn_anchor"] = anchor.astype("float32")
        input_record["rpn_cls_label"] = cls_label
        input_record["rpn_reg_target"] = reg_target
        input_record["rpn_centerness"] = centerness
        input_record["rpn_box_weight"] = box_weight
        input_record["rpn_fg_count"] = np.maximum(1., np.sum(cls_label > 0)).astype("float32")
        input_record["rpn_fg_count_reg"] = np.maximum(0.1, np.sum(centerness)).astype("float32")


if __name__ == "__main__":
    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()

        class _generate:
            def __init__(self):
                self.short = (100, 50, 25, 13, 7)
                self.long = (167, 84, 42, 21, 11)
                self.stride = (8, 16, 32, 64, 128)

            scales = (4.0,)
            aspects = (1.0,)

        class assign:
            allowed_border = -1
            top_n = 9
            mean = (.0, .0, .0, .0)
            std = (0.1, 0.1, 0.2, 0.2)

        class sample:
            image_anchor = None
            pos_fraction = None

    transform = PyramidAnchorTarget2D(AnchorTarget2DParam())
    input_record = dict()
    input_record["im_info"] = (100, 100, 1.0)
    input_record["gt_bbox"] = np.array(
        [[0, 0, 100, 100, 1], [50, 30, 60, 60, 2], [100, 50, 150, 60, 1],
         [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
        dtype=np.float32
    )
    transform.apply(input_record)
    print(input_record)
