# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.data import InstanceData
from parameterized import parameterized

from mmdet.models.roi_heads import StandardRoIHead  # noqa
from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs


def _fake_roi_head(with_shared_head=False):
    """Set a fake roi head config."""
    if not with_shared_head:
        roi_head = Config(
            dict(
                type='StandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=1,
                    featmap_strides=[4, 8, 16, 32]),
                bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=1,
                    fc_out_channels=1,
                    num_classes=4),
                mask_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=14, sampling_ratio=0),
                    out_channels=1,
                    featmap_strides=[4, 8, 16, 32]),
                mask_head=dict(
                    type='FCNMaskHead',
                    num_convs=1,
                    in_channels=1,
                    conv_out_channels=1,
                    num_classes=4),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=True,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                test_cfg=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100,
                    mask_thr_binary=0.5)))
    else:
        roi_head = Config(
            dict(
                type='StandardRoIHead',
                shared_head=dict(
                    type='ResLayer',
                    depth=50,
                    stage=3,
                    stride=2,
                    dilation=1,
                    style='caffe',
                    norm_cfg=dict(type='BN', requires_grad=False),
                    norm_eval=True),
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=14, sampling_ratio=0),
                    out_channels=1,
                    featmap_strides=[16]),
                bbox_head=dict(
                    type='BBoxHead',
                    with_avg_pool=True,
                    in_channels=2048,
                    roi_feat_size=7,
                    num_classes=4),
                mask_roi_extractor=None,
                mask_head=dict(
                    type='FCNMaskHead',
                    num_convs=0,
                    in_channels=2048,
                    conv_out_channels=1,
                    num_classes=4),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=14,
                    pos_weight=-1,
                    debug=False),
                test_cfg=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100,
                    mask_thr_binary=0.5)))
    return roi_head


def _fake_proposals(img_metas, proposal_len):
    """Create a fake proposal list."""
    results = []
    for i in range(len(img_metas)):
        result = InstanceData(metainfo=img_metas[i])
        proposal = torch.randn(proposal_len, 4).to(device='cuda')
        result.bboxes = proposal
        results.append(result)
    return results


class TestStandardRoIHead(TestCase):

    def test_init(self):
        """Test init standard RoI head."""
        # Normal Mask R-CNN RoI head
        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)
        self.assertTrue(roi_head.with_mask)

        # Mask R-CNN RoI head with shared_head
        roi_head_cfg = _fake_roi_head(with_shared_head=True)
        rpn_head = MODELS.build(roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)
        self.assertTrue(roi_head.with_mask)
        self.assertTrue(rpn_head.with_shared_head)

    @parameterized.expand([(False, ), (True, )])
    def test_standard_roi_head_loss(self, with_shared_head):
        """Tests standard roi head loss when truth is empty and non-empty."""
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        roi_head_cfg = _fake_roi_head(with_shared_head=with_shared_head)
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            if not with_shared_head:
                feats.append(
                    torch.rand(1, 1, s // (2**(i + 2)),
                               s // (2**(i + 2))).to(device='cuda'))
            else:
                feats.append(
                    torch.rand(1, 1024, s // (2**(i + 2)),
                               s // (2**(i + 2))).to(device='cuda'))
        feats = tuple(feats)

        # When truth is non-empty then both cls, box, and mask loss
        # should be nonzero for random inputs
        proposal_list = _fake_proposals(img_metas, 100)
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(s, s, 3)],
            num_items=[1],
            num_classes=4,
            with_mask=True)
        batch_data_samples = []
        for i in range(len(packed_inputs)):
            batch_data_samples.append(
                packed_inputs[i]['data_sample'].to(device='cuda'))
        out = roi_head.loss(feats, proposal_list, batch_data_samples)
        loss_cls = out['loss_cls']
        loss_bbox = out['loss_bbox']
        loss_mask = out['loss_mask']
        self.assertGreater(loss_cls.sum(), 0, 'cls loss should be non-zero')
        self.assertGreater(loss_bbox.sum(), 0, 'box loss should be non-zero')
        self.assertGreater(loss_mask.sum(), 0, 'mask loss should be non-zero')

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box and mask loss.
        proposal_list = _fake_proposals(img_metas, 100)
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(s, s, 3)],
            num_items=[0],
            num_classes=4,
            with_mask=True)
        batch_data_samples = []
        for i in range(len(packed_inputs)):
            batch_data_samples.append(
                packed_inputs[i]['data_sample'].to(device='cuda'))
        out = roi_head.loss(feats, proposal_list, batch_data_samples)
        empty_cls_loss = out['loss_cls']
        empty_bbox_loss = out['loss_bbox']
        empty_mask_loss = out['loss_mask']
        self.assertGreater(empty_cls_loss.sum(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_bbox_loss.sum(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_mask_loss.sum(), 0,
            'there should be no box loss when there are no true boxes')
