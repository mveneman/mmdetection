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


def calculate_confusion_matrix(dataset,
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
    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])

    total_iou_list = []

    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        image_iou_arrays = analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr, tp_iou_thr)

        # Add the IoU values to the total list
        for iou_list in image_iou_arrays:
            for det_label_iou_array in iou_list:
                total_iou_list.append(det_label_iou_array)

        prog_bar.update()

    mean_iou_value = mean(total_iou_list)
    return confusion_matrix, mean_iou_value


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5):
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

    true_positives = np.zeros(len(gts))
    gt_masks = []
    gt_labels = []
    for gt in gts:
        binary_mask = coordinates_to_binary_mask(gt["mask"][0])
        gt_masks.append(binary_mask)
        gt_labels.append(gt['bbox_label'])
    gt_masks = np.array(gt_masks, dtype=object)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())
    image_ious = []

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        mask_np = np.array(mask)
        det_scores = result['scores'][mask].numpy()
        # Get the detected masks; from compressed to binary masks as numpy arrays
        compressed_det_masks = [result['masks'][i] for i in range(len(mask_np)) if mask_np[i]]
        binary_det_masks = [mask_util.decode(det_mask) for det_mask in compressed_det_masks]
            
        ious = segmentation_overlaps(binary_det_masks, gt_masks)
        if ious.size != 0:
            det_label_ious = get_counting_ious(ious)
            image_ious.append(det_label_ious)

        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1

    return image_ious  


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Confusion Matrix',
                          color_theme='plasma'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(4, 4), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    int(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix_segm.png'), format='png')
    if show:
        plt.show()


def segmentation_overlaps(seg_masks1, seg_masks2, eps=1e-6):
    # Initiate iou matrix
    rows = len(seg_masks1)
    cols = len(seg_masks2)
    overlaps = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return overlaps
    
    for i in range(rows):
        for j in range(cols):
            mask1 = seg_masks1[i]
            mask2 = seg_masks2[j]

            intersection_area = np.sum(np.logical_and(mask1, mask2))
            union_area = np.sum(np.logical_or(mask1, mask2))
            union_area = np.maximum(union_area, eps)
            overlaps[i, j] = intersection_area / union_area
    return overlaps


def coordinates_to_binary_mask(mask_data):
    # Let op, hardcoded! Dit zou in theorie uit de afbeeldingsinformatie kunnen worden gehaald.
    image_size = (1024, 1024)
    binary_mask = np.zeros(image_size, dtype=np.uint8)
    seg_mask = np.array(mask_data).reshape(-1, 2)
    points = seg_mask.astype(np.int32)
    cv2.fillPoly(binary_mask, [points], color=1)
    return binary_mask


def make_metrics_json(confusion_matrix, total_mean_iou, score_thr = 0, tp_iou_thr = 0.5, save_dir=None):
    # Deze functie heb ik gemaakt voor 1 klasse (en achtergrond) en werkt dus niet voor multiclass
    true_pos = confusion_matrix[0, 0]
    false_pos = confusion_matrix[0, 1]
    false_neg = confusion_matrix[1, 0]

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    f1_score = 2 * (precision * recall)/(precision + recall)

    outfile_contents = {"IoU_treshold": tp_iou_thr,
                        "score_treshold": score_thr,
                        "precision": precision,
                        "recall": recall,
                        "F1_score": f1_score,
                        "total_mean_IoU": float(total_mean_iou)}
    
    outfile_name = "metrics.json"
    if save_dir is not None:
        outfile_name = os.path.join(save_dir, outfile_name)

    with open(outfile_name, 'w') as outfile:
        json.dump(outfile_contents, outfile)


def get_counting_ious(iou_matrix):
    # Get the maximum IoU for every segmentation
    if iou_matrix.size == 0:
        return np.zeros((0, 0))
    else:
        max_of_each_row = np.max(iou_matrix, axis = 1)
    return max_of_each_row


# Op dit moment niet gebruikt, omdat het al in één lijst wordt gegooid
def calculate_mean_iou(iou_array):
    # Concat the individual images arrays to one array
    concat_array = np.concatenate(iou_array)
    print(concat_array)

    if len(concat_array) == 0:
        total_average = 0
    else:
        total_average = np.mean(concat_array)
    return total_average


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

    confusion_matrix, total_mean_iou = calculate_confusion_matrix(dataset,results,
                                                                  args.score_thr, args.tp_iou_thr)
    
    plot_confusion_matrix(
        confusion_matrix,
        dataset.metainfo['classes'] + ('background', ),
        save_dir=args.save_dir,
        show=args.show,
        color_theme=args.color_theme)
    
    make_metrics_json(confusion_matrix, total_mean_iou, args.score_thr, args.tp_iou_thr, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
