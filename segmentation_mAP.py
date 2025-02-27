'''
URL : https://github.com/cosmiq/solaris
Code based on: https://github.com/CosmiQ/solaris/blob/main/solaris/eval/vector.py with some modifications
'''

import os
import glob
from tqdm import tqdm
import numpy as np
import geopandas as gpd


def average_score_by_class(ious, threshold=0.5):
    """ for a list of object ious by class, test if they are a counted as a
    positive or a negative.
    Arguments
    ---------
        ious : list of lists
            A list containing individual lists of ious for eachobject class.
        threshold : float , optional
            A value between 0.0 and 1.0 that determines the threshold for a true positve.
            The 0.5 is its default value , which it can be changed optionally.
    Returns
    ---------
        average_by_class : list
            A list containing the ratio of true positives for each class
    """
    binary_scoring_lists = []
    for x in ious:
        items = []
        for i in x:
            if i >= threshold:
                items.append(1)
            else:
                items.append(0)
        binary_scoring_lists.append(items)
    average_by_class = []
    for l in binary_scoring_lists:
        average_by_class.append(np.nanmean(l))
    return average_by_class




def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.

    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.

    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.

    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for _, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
        else:
            iou_score = 0
        row['iou_score'] = iou_score
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF

def get_all_objects(proposal_polygons_dir, gt_polygons_dir,
                    prediction_cat_attrib="class", gt_cat_attrib='make',
                    file_format="geojson"):
    """ Using the proposal and ground truth polygons, calculate the total.
    Filenames of predictions and ground-truth must be identical.
    unique classes present in each
    Arguments
    ---------
        proposal_polygons_dir : str
            The path that contains any model proposal polygons
        gt_polygons_dir : str
            The path that contains the ground truth polygons
        prediction_cat_attrib : str
            The column or attribute within the predictions that specifies
            unique classes
            replace "class" with the column/attribute name
        gt_cat_attrib : str 
            The column or attribute within the ground truth that
            specifies unique classes
            replace 'make' with the column/attribute name
        file_format : str , optional
            The extension or file format for predictions
    Returns
    ---------
            prop_objs : list
                All unique objects that exist in the proposals
            gt_obj : list
                All unique objects that exist in the ground truth
            all_objs : list
                A union of the prop_objs and gt_objs lists
    """
    objs = []
    os.chdir(proposal_polygons_dir)
    search = "*" + file_format
    proposal_geojsons = glob.glob(search)
    for geojson in tqdm(proposal_geojsons):
        ground_truth_poly = os.path.join(gt_polygons_dir, geojson)
        if os.path.exists(ground_truth_poly):
            ground_truth_gdf = gpd.read_file(ground_truth_poly)
            proposal_gdf = gpd.read_file(geojson)
            for index, row in (proposal_gdf.iterrows()):
                objs.append(row[prediction_cat_attrib])
    prop_objs = list(set(objs))
    os.chdir(gt_polygons_dir)
    search = "*" + file_format
    objs = []
    gt_geojsons = glob.glob(search)
    for geojson in tqdm(gt_geojsons):
        proposal_poly = os.path.join(proposal_polygons_dir, geojson)
        if os.path.exists(proposal_poly):
            proposal_gdf = gpd.read_file(proposal_poly)
            ground_truth_gdf = gpd.read_file(geojson)
            for index, row in (ground_truth_gdf.iterrows()):
                objs.append(row[gt_cat_attrib])
    gt_objs = list(set(objs))
    all_objs = gt_objs + prop_objs
    all_objs = list(set(all_objs))
    return prop_objs, gt_objs, all_objs
                        
def calculate_matching_iou(iou_GDF, proposal_row, ground_truth_gdf, prediction_cat_attrib, gt_cat_attrib):
    """
    Calculate the matching IoU for a single proposal against ground truth.

    Arguments:
    -----------
    iou_GDF : GeoDataFrame
        GeoDataFrame containing IoU scores.
    proposal_row : GeoSeries
        The row from the proposal GeoDataFrame.
    ground_truth_gdf : GeoDataFrame
        The GeoDataFrame containing ground truth polygons.
    prediction_cat_attrib : str
        The attribute in the proposal for class/category.
    gt_cat_attrib : str
        The attribute in the ground truth for class/category.

    Returns:
    --------
    float
        The IoU score if a matching ground truth is found, 0 otherwise.
    """
    if 'iou_score' in iou_GDF.columns:
        # Find the maximum IoU score and corresponding ground truth
        iou = iou_GDF.iou_score.max()
        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax()]
        id_1 = proposal_row[prediction_cat_attrib]
        id_2 = ground_truth_gdf.loc[max_iou_row.name][gt_cat_attrib]

        # Check if the class/category matches
        if id_1 == id_2:
            ground_truth_gdf.drop(max_iou_row.name, axis=0, inplace=True)
            return iou
    return 0

def process_object_category(proposal_gdf, object_category, gt_cat_attrib, confidence_attrib=None):
    """
    Process each object category for IoU and confidence score accumulation.

    Arguments:
    -----------
    proposal_gdf : GeoDataFrame
        The GeoDataFrame containing proposal polygons.
    object_category : str
        The current object category to process.
    gt_cat_attrib : str
        The attribute in the ground truth for class/category.
    confidence_attrib : str, optional
        The attribute in the proposal for confidence scores.

    Returns:
    --------
    ious : list
        List of IoU scores for the object category (all zeros in this case).
    confidences : list
        List of confidence scores for the object category.
    """
    ious = []
    confidences = []
    proposal_gdf_filtered = proposal_gdf[proposal_gdf[gt_cat_attrib] == object_category]

    for _ in range(len(proposal_gdf_filtered)):
        ious.append(0)  # Append 0 for IoU as there's no matching ground truth

    if confidence_attrib:
        confidences = [row[confidence_attrib] for _, row in proposal_gdf_filtered.iterrows()]

    return ious, confidences




def precision_calc(proposal_polygons_dir, gt_polygons_dir,
                   prediction_cat_attrib="class", gt_cat_attrib='make', confidence_attrib=None,
                   object_subset=[], threshold=0.5, file_format="geojson"):
    """ Using the proposal and ground truth polygons, calculate precision metrics.
    Filenames of predictions and ground-truth must be identical.  Will only
    calculate metric for classes that exist in the ground truth.
    Arguments
    ---------
        proposal_polygons_dir : str
            The path that contains any model proposal polygons
        gt_polygons_dir : str
            The path that contains the ground truth polygons
        prediction_cat_attrib : str
            The column or attribute within the predictions that specifies
            unique classes
            replace "class" with the name of the column/attribute
        gt_cat_attrib : str
            The column or attribute within the ground truth that
            specifies unique classes
            replace "make" with the name of the column/attribute
        confidence_attrib : str
            The column or attribute within the proposal polygons that
            specifies model confidence for each prediction
            if it exists, replace "None" with the name of this column
        object_subset : list
            A list or subset of the unique objects that are contained within the
            ground truth polygons. If empty, this will be
            auto-created using all classes that appear ground truth polygons.
        threshold : float
            A value between 0.0 and 1.0 that determines the IOU threshold for a
            true positve.
            It's set by default to 0.5.Can be changed optionally.
        file_format : str
            The extension or file format for predictions
            by default set to geojson format
    Returns
    ---------
        iou_holder : list of lists
            An iou score for each object per class (precision specific)
        precision_by_class : list
            A list containing the precision score for each class
        mPrecision : float
            The mean precision score of precision_by_class
        confidences : list of lists
            All confidences for each object for each class
    """
    ious = []
    dir = os.chdir(proposal_polygons_dir)
    search = "*" + file_format
    proposal_geojsons = glob.glob(search)
    iou_holder = []
    confidences = []
    if len(object_subset) == 0:
        prop_objs, object_subset, all_objs = get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    for i in range(len(object_subset)):
        iou_holder.append([])
        confidences.append([])

    for geojson in tqdm(proposal_geojsons):
        ground_truth_poly = os.path.join(gt_polygons_dir, geojson)
        if os.path.exists(ground_truth_poly):
            ground_truth_gdf = gpd.read_file(ground_truth_poly)
            proposal_gdf = gpd.read_file(geojson)
            i = 0
            for obj in object_subset:
                conf_holder = []
                proposal_gdf2 = proposal_gdf[proposal_gdf[prediction_cat_attrib] == obj]
                for index, row in (proposal_gdf2.iterrows()):
                    if confidence_attrib is not None:
                        conf_holder.append(row[confidence_attrib])
                    iou_GDF = calculate_iou(row.geometry, ground_truth_gdf)
                    iou = calculate_matching_iou(iou_GDF, row, ground_truth_gdf, prediction_cat_attrib, gt_cat_attrib)
                    ious.append(iou)
                for item in ious:
                    iou_holder[i].append(item)
                if confidence_attrib is not None:
                    for conf in conf_holder:
                        confidences[i].append(conf)
                ious = []
                i += 1
        else:
            print("Warning- No ground truth for:", geojson)
            proposal_gdf = gpd.read_file(geojson)
            i = 0

            for j, obj in enumerate(object_subset):
                ious, conf_holder = process_object_category(proposal_gdf, obj, gt_cat_attrib, confidence_attrib)
                iou_holder[j].extend(ious)
                confidences[j].extend(conf_holder)
                i += 1
                ious = []
    precision_by_class = average_score_by_class(iou_holder, threshold=threshold)
    precision_by_class = list(np.nan_to_num(precision_by_class))
    mPrecision = np.nanmean(precision_by_class)
    print("mPrecision:", mPrecision)

    return iou_holder, precision_by_class, mPrecision, confidences

