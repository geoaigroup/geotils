import numpy as np
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import json
import glob
import tmp as utils
import pandas as pd

def iou_numpy(outputs, labels):
    """
    Calculate the Intersection over Union (IoU) score between binary segmentation outputs and ground truth labels.

    Args:
        outputs (torch.Tensor): Binary segmentation outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: IoU score.
    """
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score


class MatchingAlgorithm():
    """
    Initialize the MatchingAlgorithm.
    This class is designed to perform matching between ground truth bounding boxes
    and predicted bounding boxes based on Intersection over Union (IoU) scores.
    It takes lists of ground truth (GT) and predicted bounding boxes along with
    an optional IoU threshold for matching.

    Args:
        gt_bbox (List[np.ndarray]): List of ground truth bounding boxes.
        pred_bbox (List[np.ndarray]): List of predicted bounding boxes.
        iou_threshold (float): Intersection over Union (IoU) threshold for matching. Default is 0.5.
    """
    def __init__(self, gt_bbox, pred_bbox, iou_threshold=0.5):
        self.gt_bboxes = gt_bbox
        self.pred_bboxes = pred_bbox
        self.iou_threshold = iou_threshold

    def matching(self):
        """ 
        This method calculates the IoU scores between all pairs of ground truth
        and predicted bounding boxes. It then identifies true positives, false positives,
        and false negatives based on the IoU threshold. The method returns various metrics
        including IoU list, F1 scores, indices of true positives, false positives, and false negatives.

        Returns:
            Tuple of results including IoU list, F1 scores, indices of true positives,
            indices of false positives, indices of false negatives, matching scores,
            precision, and recall.
        """
        if len(self.pred_bboxes) == 0 or len(self.gt_bboxes) == 0:
            print("Both predicted and ground truth bounding boxes are empty.")
            return [], [], [], [], [], [], [], [], []

        iou_matrix = np.zeros((len(self.pred_bboxes), len(self.gt_bboxes)))

        for i in range(len(self.pred_bboxes)):
            for j in range(len(self.gt_bboxes)):
                iou_matrix[i, j] = iou_numpy(torch.from_numpy(self.pred_bboxes[i]), torch.from_numpy(self.gt_bboxes[j]))

        iou_list = []
        f1_scores = []
        pred_matched = set()
        gt_matched = set()
        tp_pred_indices = []
        tp_gt_indices = []
        m_score=[]
        mscores=[]

        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            max_index = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape)
            iou_list.append(max_iou)
            pred_matched.add(max_index[0])
            gt_matched.add(max_index[1])

            tp_pred_indices.append(max_index[0])
            tp_gt_indices.append(max_index[1])

            f1_score = 2 * max_iou / (max_iou + 1)
            f1_scores.append(f1_score)

            print(
                f"Matched predicted box {max_index[0]} with GT box {max_index[1]}, IoU = {max_iou}, F1 = {f1_score}")
            m_score={
                'pred_box':int(max_index[0]),
                'GT_box':int(max_index[1]),
                'iou':float(max_iou),
                'f1':float(f1_score)
            }
            mscores.append(m_score)
            iou_matrix[max_index[0], :] = 0
            iou_matrix[:, max_index[1]] = 0

        for i in set(range(len(self.pred_bboxes))) - pred_matched:
            iou_list.append(0)
            f1_scores.append(0)
            print(f"Unmatched predicted box {i} has no match, IoU = 0, F1 = 0")

        for i in set(range(len(self.gt_bboxes))) - gt_matched:
            iou_list.append(0)
            f1_scores.append(0)
            print(f"Unmatched GT box {i} has no match, IoU = 0, F1 = 0")

        print("number of GT boxes:", len(self.gt_bboxes))
        print("number of predicted boxes:", len(self.pred_bboxes))

        fp_indices = list(set(range(len(self.pred_bboxes))) - pred_matched)
        fn_indices = list(set(range(len(self.gt_bboxes))) - gt_matched)

        true_positives = len(tp_pred_indices)
        false_positives = len(fp_indices)
        false_negatives = len(fn_indices)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return iou_list, f1_scores, tp_pred_indices, tp_gt_indices, fp_indices, fn_indices, mscores, precision, recall

    def tp_iou(self, tp_pred_indices, tp_gt_indices):
        """
        Calculate IoU for true positive matches by taking the indices of true positive predicted and 
        ground truth boxes and calculating the Intersection over Union (IoU) scores for each true positive match.
        It returns a tuple containing the list of IoU scores for true positive matches
        and the average IoU.

        Args:
            tp_pred_indices (List[int]): Indices of true positive predicted boxes.
            tp_gt_indices (List[int]): Indices of true positive ground truth boxes.

        Returns:
            Tuple containing the list of IoU scores for true positive matches and the average IoU.
        """
        tp_iou_list = []
        for i, j in zip(tp_pred_indices, tp_gt_indices):
            iou = iou_numpy(torch.from_numpy(self.pred_bboxes[i]), torch.from_numpy(self.gt_bboxes[j]))
            tp_iou_list.append(float(np.nan_to_num(iou)))

        if len(tp_iou_list) > 0:
            avg_tp_iou = float(np.mean(tp_iou_list))
        else:
            avg_tp_iou = None
        return tp_iou_list, avg_tp_iou


class CalScores:
    """
    Initialize the CalScores class.
    Designed for evaluating segmentation results, particularly for building detection tasks. It provides 
    methods to calculate both micro-level and macro-level matching metrics between predicted and 
    ground truth building masks. Additionally, it facilitates the storage and visualization of results, 
    such as saving score files and generating annotated images.

    Args:
        output_dir (str): Directory to save output files.
        score_dir (str): Directory to save score files.
    """
    def __init__(self,output_dir,score_dir):
        self.output_dir=output_dir
        self.score_dir=score_dir

    def micro_match_iou(self,pred_mask, name, gt, image,input_point,input_label,tile_boxes, save=None, visualize=None):
        """
        Calculate micro-level matching metrics and save results.
        Responsible for calculating micro-level matching metrics between predicted and ground truth 
        building masks for a specific image tile. It takes into account various metrics such as IoU 
        (Intersection over Union), F1 score, precision, and recall. The results are stored in a JSON file, 
        and an annotated image is generated for visual inspection.

        Args:
            pred_mask (torch.Tensor): Predicted mask.
            name (str): Name identifier for the image.
            gt (dict): Ground truth information.
            image (numpy.ndarray): Input image.
            input_point: Input points.
            input_label: Input labels.
            tile_boxes: Boxes related to tiles.
            save: Flag set for saving.
            visualize: Flag set for visualization.
        """
        pred_tile = []
        gt_tile = []
        msk = pred_mask.int()
        msk = msk.cpu().numpy()
        scores_b = []
        score = {}
        mask_tile = np.zeros(image.shape[:2])

        for i in range(msk.shape[0]):
            batch = msk[i]
            for b in range(batch.shape[0]):
                mask_tile = mask_tile + batch[b]
                pred_tile.append(batch[b])


        gt_tile=utils.convert_polygon_to_mask_batch(gt['geometry'])

        matcher = MatchingAlgorithm(gt_tile, pred_tile)
        iou_list, f1_scores, tp_pred_indices, tp_gt_indices, fp_indices, fn_indices, mscores, precision,recall = matcher.matching()
        tp_iou_list, avg_tp_iou = matcher.tp_iou(tp_pred_indices, tp_gt_indices)

        score['iou_list'] = iou_list
        score['f1_scores'] = f1_scores
        score['tp_iou_list'] = tp_iou_list
        score['fp_indices'] = fp_indices
        score['fn_indices'] = fn_indices
        score['Mean_iou'] = np.mean(iou_list, dtype=float)
        score['Mean_f1'] = np.mean(f1_scores, dtype=float)
        score['avg_tp_iou'] = float(avg_tp_iou) if avg_tp_iou is not None else 0.0
        score['precision'] = precision
        score['recall'] = recall

        for s in mscores:
            scores_b.append(s)
        scores_b.append(score)

        with open(self.score_dir + f'/{name}_score.json', 'w') as f1:
            json.dump(scores_b, f1)
            
        if save is not None:
            utils.save_shp(pred_mask,name,self.output_dir,image.shape[:2])
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        utils.show_mask(mask_tile, plt.gca(), random_color=False)
        if input_point is not None:
            utils.show_points(input_point.cpu(), input_label.cpu(), plt.gca())
        for box in tile_boxes:
            utils.show_box(box,plt.gca())
        
        if visualize is not None:
            for box in tile_boxes:
                x = []
                y = []
                for i in range(len(box)):
                    if i % 2 == 0:
                        x.append(box[i])
                    else:
                        y.append(box[i])
                plt.plot(x, y)
        plt.show()

        if visualize is not None:
            gtmask=np.zeros((384,384))
            for g in gt_tile:
                gtmask=g+gtmask
            plt.imshow(gtmask)
            plt.show()
        

    def macro_score(self):
        """
        Calculate macro-level matching metrics and save results.
        Aggregates and analyzes the macro-level matching metrics for all image tiles. It reads previously 
        generated score files, calculates average scores, and generates a CSV file summarizing the overall performance. 
        This method provides insights into the global performance of the segmentation model across multiple tiles.
        """
        score_list = []
        all_iou_scores = []
        all_f1_scores = []
        all_precision=[]
        all_recall=[]
        for i in glob.glob(os.path.join(self.score_dir, "*.json")):
            name = i.split("/")[-1]
            name = name.split("_score")[0]

            f = open(i)
            file_data = json.load(f)
            ds = {}
            iou = file_data[len(file_data) - 1]["Mean_iou"]
            f1 = file_data[len(file_data) - 1]["Mean_f1"]
            avg_tp_iou = file_data[len(file_data) - 1]["avg_tp_iou"]
            precision = file_data[len(file_data) - 1]["precision"]
            recall = file_data[len(file_data) - 1]["recall"]

            all_precision.append(precision)
            all_recall.append(recall)

            all_iou_scores.append(file_data[len(file_data) - 1]["iou_list"])
            all_f1_scores.append(file_data[len(file_data) - 1]["f1_scores"])


            ds["name"] = name
            ds["iou"] = iou
            ds["f1"] = f1
            ds["avg_tp_iou"] = avg_tp_iou
            ds["precision"] = precision
            ds["recall"] = recall
            score_list.append(ds)

        df = pd.DataFrame(score_list)
        df.to_csv(self.score_dir + "/scores.csv", index=False)

        all_i = []
        all_f = []
        all_tpi = []
        all_tpf = []

        for i1, f11 in zip(all_iou_scores, all_f1_scores):
            for i2, f12 in zip(i1, f11):
                all_i.append(i2)
                all_f.append(f12)
                if i2 > 0 and f12 > 0:
                    all_tpi.append(i2)
                    all_tpf.append(f12)

        total_iou = np.nanmean(np.array(all_i))
        total_f1 = np.nanmean(np.array(all_f))
        total_tpiou = np.mean(np.array(all_tpi))
        total_tpf1 = np.mean(np.array(all_tpf))
        total_precision = np.mean(np.array(all_precision))
        total_recall = np.mean(np.array(all_recall))

        print("Mean iou score of all buildings in all tiles:", total_iou)
        print("Mean F1 score of all buildings in all tiles:", total_f1)
        print("Mean tp iou score of all buildings in all tiles:", total_tpiou)
        print("Mean tp f1 score of all buildings in all tiles:", total_tpf1)
 
