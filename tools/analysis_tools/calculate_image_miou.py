import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root

import pycocotools.mask as mask_util
import cv2
import json
from statistics import mean

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_image_level_mIoU(dataset,
                               results,
                               score_thr=0,
                               tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """

    total_iou_list = []

    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        image_iou= analyze_per_img_dets(gts, res_bboxes)
        total_iou_list.append(image_iou)
        prog_bar.update()

    mean_iou_value = mean(total_iou_list)
    return mean_iou_value


def analyze_per_img_dets(gts, result):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    gt_masks = []
    for gt in gts:
        binary_mask = coordinates_to_binary_mask(gt["mask"][0])
        gt_masks.append(np.array(binary_mask))
    
    binary_det_masks = [mask_util.decode(instance) for instance in result['masks']]
    
    if len(gt_masks) == 0 and len(binary_det_masks) == 0:
        return 1
    elif len(gt_masks) == 0 or len(binary_det_masks) == 0:
        return 0
    
    merged_gt_mask = merge_binary_masks(gt_masks)
    merged_mask = merge_binary_masks(binary_det_masks)

    iou = image_segmentation_overlaps(merged_mask, merged_gt_mask)
    return iou  


def merge_binary_masks(binary_mask_list):
    """
    Merges a list which contains multiple binary masks into one masks
    Input is a list, in which each individual mask is a array
    """
    # Create empty mask
    merged_mask = np.zeros_like(binary_mask_list[0])
    # Iterate trough every mask
    for binary_mask in binary_mask_list:
        merged_mask |= binary_mask
    return merged_mask


def image_segmentation_overlaps(mask1, mask2, eps=1e-6):
    intersection_area = np.sum(np.logical_and(mask1, mask2))
    union_area = np.sum(np.logical_or(mask1, mask2))
    union_area = np.maximum(union_area, eps)
    iou = intersection_area / union_area
    return iou


def coordinates_to_binary_mask(mask_data):
    # Let op, hardcoded! Dit zou in theorie uit de afbeeldingsinformatie kunnen worden gehaald.
    image_size = (1024, 1024)
    binary_mask = np.zeros(image_size, dtype=np.uint8)
    seg_mask = np.array(mask_data).reshape(-1, 2)
    points = seg_mask.astype(np.int32)
    cv2.fillPoly(binary_mask, [points], color=1)
    return binary_mask


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    total_mean_iou = calculate_image_level_mIoU(dataset, results, args.score_thr, args.tp_iou_thr)
    print("Image level mean IoU: ", total_mean_iou)

if __name__ == '__main__':
    main()
