import sys 
sys.path.append("/home/giacomo/projects/P0_YOLOcate/ultralytics/utils")

import json
from pathlib import Path
import numpy as np
import pandas as pd

from metrics import ConfusionMatrix



def get_test_counts(ann_file: str, class_ids: list) -> None:
    """
    From a test set annotations file as produced by preprocessing.py, collect the class counts for 
    each image in the test set. 
    Arguments: 
        ann_file (str):     path to the annotations file. Is expected to contain a dictionary where 
                            key = image id / filename and value = list of annotations (which themselves 
                            are dictionaries).
        class_ids (list):   list of integer class ids. 
    Returns: 
        None 
    """
    
    with open(ann_file, "r") as f:
        ann_dict = json.load(f)

    all_imgs = {} 
    
    counts_dir = f"{Path(ann_file).parent}/image_counts"
    Path(counts_dir).mkdir()   

    for img, ann_list in ann_dict.items():
        img_counts = {cls_id: 0 for cls_id in class_ids}

        # collect class counts for current image
        for ann in ann_list: 
            img_counts[ann["category_id"]] += 1
        
        assert img not in all_imgs.keys()
        all_imgs[img] = img_counts

        # write image counts to file
        json_fn = f"{counts_dir}/{img}.json"
        with open(json_fn, "w") as f:
            json.dump(img_counts, f, indent=1)

    
    # collect image counts to get total count
    total_counts = {cls_id: 0 for cls_id in class_ids}
    
    for img in all_imgs.keys():
        for cls_id in total_counts.keys():
            total_counts[cls_id] += all_imgs[img][cls_id]

    total_counts_fn = f"{Path(ann_file).parent}/counts_total.json"
    with open(total_counts_fn, "w") as f:
        json.dump(total_counts, f, indent=1)


    

def compute_errors(gt_counts_dir: str, pred_counts_dir: str, class_ids: list, output_dir: str) -> dict:
    """
    Given directories containing .json files of ground truth counts of an image set, and predicted counts, 
    compute the MAE and MSE. Also outputs an excel sheet containing the count differences per image.
    Arguments:
        gt_counts_dir (str):        path to the directory containing the gt counts.
        pred_counts_dir (str):      path to the directory containing the predicted counts.
        class_ids (list):           list of integer class ids. 
        output_dir (str):           path to a folder where the output can be stored.
    Returns:
        dictionary containing the MAE and MSE per class. 
    """

    # get paths to prediction files
    gt_files = [str(path) for path in Path(gt_counts_dir).rglob("*.json")]
    pred_files = [str(path) for path in Path(pred_counts_dir).rglob("*.json")]

    # make empty df to collect count differences
    df_cols = ["fn"]
    df_cols.extend([str(cls_id) for cls_id in class_ids])
    count_diffs_df = pd.DataFrame(columns=df_cols)

    for fn_gt in gt_files:
        
        # get matching prediction file for a gt file 
        search_str = fn_gt.split("/")[-1].split(".")[0]
        fn_pred = [pf for pf in pred_files if search_str in pf]

        fn_pred = fn_pred[0]

        with open(fn_gt, "r") as gt:
            gt_dict = json.load(gt)

        with open(fn_pred, "r") as pred:
            pred_dict = json.load(pred)

        # if no predictions for a class were made, the count number is zero 
        pred_dict_updated = {cls_id: 0 if str(cls_id) not in pred_dict.keys() else pred_dict[str(cls_id)] for cls_id in class_ids }

        # get difference an dput into df
        diffs = {str(cls_id): [gt_dict[str(cls_id)] - pred_dict_updated[cls_id]] for cls_id in class_ids}
        diffs["fn"] = [search_str]
        row_df = pd.DataFrame(diffs)
        count_diffs_df = pd.concat([count_diffs_df, row_df], ignore_index=True)

    # get overall MAE and MSE per class 
    summary_dict = {cls_id: {"MAE": count_diffs_df[f"{cls_id}"].abs().sum() / count_diffs_df.shape[0],
                             "MSE": (count_diffs_df[f"{cls_id}"] ** 2).sum() / count_diffs_df.shape[0]} for cls_id in class_ids}
    
    # output
    with open(f"{output_dir}/errors.json", "w") as f:
        json.dump(summary_dict, f, indent=1)

    count_diffs_df.to_excel(f"{output_dir}/count_diffs.xlsx")

    return summary_dict



def get_eval_metrics(dets_dir: str, class_ids: list, class_names: list, task: str, output_dir: str = None, k: int = 3) -> None:
    """
    Aggregate image level confusion matrices  into a global confusion matrix. Also generates a .csv file that for each image 
    contains the evaluation metrics per class, and a .json file that for each metric contains the highests scoring filename.
    Arguments: 
        dets_dir (str):         path to the directory containing the image level confusion matrices
        class_ids (list):       list of class ids
        class_names (list):     class names that match the ids
        task (str):             task string. can be 'detect' or 'locate'.
        output_dir (str):       path to a folder where the method's output can be stored. 
        k (int):                indicates the number of images to select per evaluation metric. E.g., when k=3, 
                                the 3 highest scoring images per evaluation metric will be selected and put into 
                                the output. 
    """

    # initilaize the global confusion matrix
    cfm_glob = np.zeros((len(class_ids) + 1, len(class_ids) + 1))
    # initialize the global data frame
    df_columns = [[f"{cls_id}_tp", f"{cls_id}_fp", f"{cls_id}_fn", f"{cls_id}_total"] for cls_id in class_ids]
    df_columns.append(["backgorund_tp", "background_fp", "backgorund_fn", "background_total"])
    df_columns_flat = ["fn"]
    df_columns_flat.extend([col for class_cols in df_columns for col in class_cols])
    preds_df = pd.DataFrame(columns=df_columns_flat)

    # aggregate image level matrices
    for cfm in Path(dets_dir).rglob("*.npy"):
        cfm_img = np.load(cfm)

        # accumulate global cfm
        cfm_glob += cfm_img
        
        # copied from ultralytics repo
        tp = cfm_img.diagonal()
        fp = cfm_img.sum(1) - tp 
        fn = cfm_img.sum(0) - tp
        total = tp + fp 

        # assemble dict to update dataframe
        row_dict = {"fn": "_".join(str(cfm.stem).split("_")[:-1])}

        for cls_id in class_ids:
            cls_dict = {f"{cls_id}_tp": [tp[cls_id]], f"{cls_id}_fp": [fp[cls_id]], f"{cls_id}_fn": [fn[cls_id]], f"{cls_id}_total": [total[cls_id]]}
            row_dict.update(cls_dict)

        bg_dict = {f"background_tp": [tp[len(class_ids)]], 
                   f"background_fp": [fp[len(class_ids)]], 
                   f"background_fn": [fn[len(class_ids)]], 
                   f"background_total": [total[len(class_ids)]]}      

        row_dict.update(bg_dict)  

        row_df = pd.DataFrame(row_dict)
        preds_df = pd.concat([preds_df, row_df], ignore_index=True)

    output_path = output_dir if output_dir else Path(dets_dir).parent

    preds_df.to_excel(f"{output_path}/pred_stats_ovrall.xlsx")

    cfm_obj = ConfusionMatrix(nc=len(class_ids), task=task)
    cfm_obj.matrix = cfm_glob

    cfm_obj.plot(save_dir=output_path, names=class_names)
    cfm_obj.plot(normalize=False, save_dir=output_path, names=class_names)

    topk_dict = {}

    for col in list(preds_df.columns):
        if col != "fn":
            topk = preds_df.sort_values(by=col, ascending=False).head(k)
            fns = topk["fn"].to_list()
            topk_dict[col] = fns

    with open(f"{output_path}/topk_by_metric.json", "w") as f:
        json.dump(topk_dict, f)

