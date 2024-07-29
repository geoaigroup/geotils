from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
from tqdm import tqdm

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)
    iou = I / (U + 1e-9)
    return 1.0 if U == 0 else iou

def compute_IoU_cIoU(input_json, gti_annotations):
    # Load ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Load predictions
    with open(input_json, 'r') as f:
        predictions = json.load(f)

    # Validate predictions
    image_ids_gt = set(coco_gti.getImgIds())
    predictions = [p for p in predictions if p['image_id'] in image_ids_gt]

    # Filtered predictions dictionary
    predictions_dict = {p['image_id']: [] for p in predictions}
    for p in predictions:
        predictions_dict[p['image_id']].append(p)

    image_ids = coco_gti.getImgIds(catIds=coco_gti.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []

    for image_id in bar:
        img = coco_gti.loadImgs(image_id)[0]

        if image_id not in predictions_dict:
            continue

        anns_pred = predictions_dict[image_id]
        mask_pred = np.zeros((img['height'], img['width']), dtype=bool)
        N_pred = 0
        for ann in anns_pred:
            if 'segmentation' not in ann or len(ann['segmentation']) == 0:
                continue
            rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            mask_pred = mask_pred | (m > 0)
            N_pred += len(ann['segmentation'][0]) // 2

        ann_ids_gt = coco_gti.getAnnIds(imgIds=img['id'])
        anns_gt = coco_gti.loadAnns(ann_ids_gt)
        mask_gt = np.zeros((img['height'], img['width']), dtype=bool)
        N_gt = 0
        for ann in anns_gt:
            if 'segmentation' not in ann or len(ann['segmentation']) == 0:
                continue
            rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            mask_gt = mask_gt | (m > 0)
            N_gt += len(ann['segmentation'][0]) // 2

        if np.sum(mask_pred) == 0 or np.sum(mask_gt) == 0:
            print(f"Skipping image {image_id} due to empty mask.")
            continue

        ps = 1 - np.abs(N_pred - N_gt) / (N_pred + N_gt + 1e-9)
        iou = calc_IoU(mask_pred, mask_gt)
        list_iou.append(iou)
        list_ciou.append(iou * ps)

        bar.set_description("iou: %2.4f, c-iou: %2.4f" % (np.mean(list_iou), np.mean(list_ciou)))
        bar.refresh()

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))
