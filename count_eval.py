import json
from pathlib import Path
import pandas as pd


def get_counts_from_COCO_izembek(ann_file: str, class_ids: list) -> None:
    
    with open(ann_file, "r") as f:
        ann_dict = json.load(f)

    all_imgs = {} 
    
    counts_dir = f"{Path(ann_file).parent}/image_counts"
    Path(counts_dir).mkdir()   

    for img, ann_list in ann_dict.items():
        img_counts = {cls_id: 0 for cls_id in class_ids}

        for ann in ann_list: 
            img_counts[ann["category_id"]] += 1
        
        assert img not in all_imgs.keys()
        all_imgs[img] = img_counts

        json_fn = f"{counts_dir}/{img}.json"
        with open(json_fn, "w") as f:
            json.dump(img_counts, f, indent=1)

    
    total_counts = {cls_id: 0 for cls_id in class_ids}
    
    for img in all_imgs.keys():
        for cls_id in total_counts.keys():
            total_counts[cls_id] += all_imgs[img][cls_id]

    total_counts_fn = f"{Path(ann_file).parent}/counts_total.json"
    with open(total_counts_fn, "w") as f:
        json.dump(total_counts, f, indent=1)


def get_total_counts_SAHI(sahi_results: str, class_ids: list) -> None:
    with open(sahi_results, "r") as f:
        pred_dict = json.load(f)

    total_counts = {cls_id: 0 for cls_id in class_ids}

    for pred in pred_dict:
        total_counts[pred["category_id"]] += 1
    
    output_fn = f"{Path(sahi_results).parent}/counts_total.json"
    with open(output_fn, "w") as f:
        json.dump(total_counts, f, indent=1)

    

def compute_errors(gt_counts_dir: str, pred_counts_dir: str, class_ids: list, output_dir: str) -> dict:
    gt_files = [str(path) for path in Path(gt_counts_dir).rglob("*.json")]
    pred_files = [str(path) for path in Path(pred_counts_dir).rglob("*.json")]

    df_cols = ["fn"]
    df_cols.extend([str(cls_id) for cls_id in class_ids])
    count_diffs_df = pd.DataFrame(columns=df_cols)

    for fn_gt in gt_files:
        if "total" in fn_gt:
            continue
        
        search_str = fn_gt.split("/")[-1].split(".")[0]
        fn_pred = [pf for pf in pred_files if search_str in pf]
        
        assert len(fn_pred) == 1

        fn_pred = fn_pred[0]

        with open(fn_gt, "r") as gt:
            gt_dict = json.load(gt)

        with open(fn_pred, "r") as pred:
            pred_dict = json.load(pred)

        pred_dict_updated = {cls_id: 0 if str(cls_id) not in pred_dict.keys() else pred_dict[str(cls_id)] for cls_id in class_ids }

        diffs = {str(cls_id): [gt_dict[str(cls_id)] - pred_dict_updated[cls_id]] for cls_id in class_ids}

        diffs["fn"] = [search_str]

        row_df = pd.DataFrame(diffs)

        count_diffs_df = pd.concat([count_diffs_df, row_df], ignore_index=True)

    summary_dict = {cls_id: {"MAE": count_diffs_df[f"{cls_id}"].abs().sum() / count_diffs_df.shape[0],
                             "MSE": (count_diffs_df[f"{cls_id}"] ** 2).sum() / count_diffs_df.shape[0]} for cls_id in class_ids}
    
    with open(f"{output_dir}/errors.json", "w") as f:
        json.dump(summary_dict, f, indent=1)

    count_diffs_df.to_excel(f"{output_dir}/count_diffs.xlsx")

    return summary_dict

    








