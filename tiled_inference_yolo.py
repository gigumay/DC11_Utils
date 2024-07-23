import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")
sys.path.append("/home/giacomo/projects/P0_YOLOcate/ultralytics/utils")

import shutil
import tqdm
import json
import random
import math

import torch
import torchvision
import cv2
import pandas as pd
import visualization_utils as vis_utils
import numpy as np

from pathlib import Path

from ultralytics import YOLO
from ops import loc_nms, generate_radii_t
from metrics import ConfusionMatrix

from processing_utils import *
from globs import *

PT_VIS_RADIUS = 3



def load_img_gt(annotations: dict,  ann_format: str, task: str, device=torch.device, box_dims: dict = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expects the annotations in an image as a dictionary of dictionaries, and from that creates tensors for the ground truth labels. 
    Arguments:
        annotations (dictionary):           dictionary containing all annotations (as dictionaries) within the image under consideration.
        ann_format (string):                string indicatin the annotation format
        task (string):                      string indicating the prediction task. Can either be 'detect' or 'locate'
        device (torch.device):              device on which to store the gt tensors
        box_dims (dict):                    dimensions of ground-truth bounding boxes
    Returns:
        tuple[torch.Tensor, torch.Tensor]:  the ground truth coordinates and classes
    """
    coords_list = []
    cls_list = []

    for ann in annotations:
        label = ann[DATA_ANN_FORMATS[ann_format]["label_key"]]
        x_center = label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["width_idx"]] / 2.0)
        y_center = label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["height_idx"]] / 2.0)

        box_dims_checked = {}

        if box_dims:
            box_dims_checked["width"] = box_dims["width"]
            box_dims_checked["height"] = box_dims["height"]
        else: 
            box_dims_checked["width"] = label[DATA_ANN_FORMATS[ann_format]["width_idx"]]
            box_dims_checked["height"] = label[DATA_ANN_FORMATS[ann_format]["height_idx"]]

        if task == "detect":
            xmin = x_center - (box_dims_checked["width"]/2.0)
            xmax = x_center + (box_dims_checked["width"]/2.0)
            ymin = y_center - (box_dims_checked["height"]/2.0)
            ymax = y_center + (box_dims_checked["height"]/2.0)
            coords_list.append([xmin, ymin, xmax, ymax])
        else:
            coords_list.append([x_center, y_center])
    
        cls_list.append(ann["category_id"])


    coords_t = torch.tensor(coords_list, device=device)
    
    # sanity check 
    assert coords_t.shape[1] == len(coords_list[0])

    cls_t = torch.tensor(cls_list, device=device)

    return coords_t, cls_t 


def collect_boxes(predictions: list, class_ids: list, patches: list, device: torch.device, patch_output_dir: tuple[str, None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all patch-level boxes from a list of predictions and map them to the image space. If an output directory 
    is specified, patch level counts will be writtento a .json file in that directory. 
    Arguments:
        predictions (list):         list containing all patch predictions.
        class_ids (list):           list of class ids. 
        patches (list):             list containing all patches.
        device (torch.device):      device used for inference.
        patch_output_dir (str):     path to the folder where patch-level counts can be stored. 
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the coordinates, confidence scores and classes of the image level predictions

    """

    all_preds = torch.empty((0, 6), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.boxes

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordinates, confidence and class into one tensor
        data_merged = torch.hstack((data.xyxy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # write patch level predictions to file
        if patch_output_dir:
            patch_count = {cls_id: 0 for cls_id in class_ids}
            pred_counts = torch.bincount(data.cls)

            for i in range(pred_counts.size(dim=0)):
                patch_count[i] = pred_counts[i]

            patch_coords_str = f"{patches[i]['coords']['x_min']}_{patches[i]['coords']['y_min']}"
            with open(f"{patch_output_dir}/{patch_coords_str}.json", "w") as f:
                json.dump(patch_count, f)

        # map prediction to image_level
        data_merged[:, [0, 2]] = data_merged[:, [0,2]] + patches[i]["coords"]["x_min"]
        data_merged[:, [1, 3]] = data_merged[:, [1,3]] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((4, 1, 1), 1)


def collect_locations(predictions: list, class_ids: list, patches: list, device: torch.device, patch_output_dir: tuple[str, None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all patch-level locations from a list of predictions and map them to the image space. If an output directory 
    is specified, patch level counts will be writtento a .json file in that directory. 
    Arguments:
        predictions (list):         list containing all patch predictions.
        class_ids (list):           list of class ids. 
        patches (list):             list containing all patches.
        device (torch.device):      device used for inference.
        patch_output_dir (str):     path to the folder where patch-level counts can be stored. 
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the coordinates, confidence scores and classes of the image level predictions

    """

    all_preds = torch.empty((0, 4), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.locations

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordiantes, confidence and class into one tensor
        data_merged = torch.hstack((data.xy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # write patch level predictions to file
        if patch_output_dir:
            patch_count = {cls_id: 0 for cls_id in class_ids}
            pred_counts = torch.bincount(data.cls.int())

            for i in range(pred_counts.size(dim=0)):
                patch_count[i] = pred_counts[i].item()

            patch_coords_str = f"{patches[i]['coords']['x_min']}_{patches[i]['coords']['y_min']}"
            with open(f"{patch_output_dir}/{patch_coords_str}.json", "w") as f:
                json.dump(patch_count, f)

        # map prediction to image_level
        data_merged[:, 0] = data_merged[:, 0] + patches[i]["coords"]["x_min"]
        data_merged[:,1] = data_merged[:, 1] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image level tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((2, 1, 1), 1)


def collect_predictions_wrapper(task: str, predictions: list, patches: list, device: torch.device, patch_output_dir: tuple[str, None],
                                class_ids: list = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper for 'collect_boxes' and 'collect_locations'. 
    Arguments:
        task (str):                 string indicating the model task. Can be 'detect' or 'locate'.
        predictions (list):         list containing all predictions made on an image's patches.
        patches (list):             list containing all patches of an image.
        device (torch.device):      device used for inference.
        patch_output_dir (str):     path to the folder where patch-level predictions can be stored. 
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the coordinates, confidence scores and classes of the image level predictions

    """
    if task == "detect":
        return collect_boxes(predictions=predictions, patches=patches, device=device, patch_output_dir=patch_output_dir)
    else:
        return collect_locations(predictions=predictions, class_ids=class_ids, patches=patches, device=device, patch_output_dir=patch_output_dir)
    

def plot_annotated_img(img_fn: str, coords: torch.Tensor, cls: torch.Tensor, output_dir: str, pre_nms: bool = False,) -> None:
    """
    Plot an image and visualize the corresponding predictions.
    Arguments:
        img_fn (str):               path to the imnage file
        coords (torch.Tensor):      tensor containing the prediction coordinates. Either 4 (boxes in xyxy format) or two (points) columns. 
        cls (torch.Tensor):         tensor containing the predicted classes. 
        output_dir (str):           path to the folder where the plot can be stored.
        pre_nms (bool):             if True, the coords tensor is assumed to contain predictions before global NMS was applied.
    """

    img_arr = cv2.imread(img_fn)
    boxes = coords.shape[1] == 4

    for i in range(coords.shape[0]):
        if boxes:
            cv2.rectangle(img=img_arr, pt1=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                          pt2=(int(round(coords[i, 2].item())), int(round(coords[i, 3].item()))), color=CLASS_COLORS[int(cls[i].item())], 
                          thickness=1)
        else:
            cv2.circle(img=img_arr, center=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                       radius=PT_VIS_RADIUS, color=CLASS_COLORS[int(cls[i].item())], thickness=-1)

    output_ext = "_pre_nms" if pre_nms else ""
            
    assert cv2.imwrite(f"{output_dir}/{Path(img_fn).stem}{output_ext}.jpg", img_arr), "Plotting failed!"


def run_tiled_inference(model_file: str, task: str, class_ids: list, imgs_dir: str, img_files_ext: str, patch_dims: dict, 
                        patch_overlap: float, output_dir: str, iou_thresh: float = None, dor_thresh: float = None, radii: dict = None, 
                        ann_file: str = None, ann_format: str = None, box_dims: dict = None, vis_prob: float = -1.0, vis_density: int = math.inf, 
                        patch_quality: int = 95, save_pre_output: bool = False, rm_tiles: bool = True, save_patch_data: bool = False, 
                        verbose: bool = False) -> None:
    """
    Perform tiled inference on a directory of images. If ground truth annotations are available, a confusion matrix is produced 
    for each image.
    Arguments:
        model_file (str):           path to the model file (.pt) to be used for inference.
        task (str):                 task string. Can be 'detect' or 'locate'.
        class_ids (list):           list of integer class IDs in the data the model was trained on.
        imgs_dir (str):             path to the folder containing the images inference needs to be performed on.
        img_files_ext (str):        Extension of image files. E.g., 'JPG'.
        patch_dims (dict):          dictionary with keys 'width' & 'height' specifying the tile dimensions. 
        patch_overlap (float):      amount of overlap (fraction) between tiles. 
        output_dir (str):           path to the directory the output will be stored in. 
        iou_thresh (float):         IoU threshold to use during NMS when stitching tiles together. 
        dor_thresh (float):         DoR threshold to use during NMS when stitching tiles together. 
        radii (dict):               dictionary containing the radius values for localization NMS.
        ann_file (string):          path to the file containing the annotations for the images to be processed. The file is
                                    expected to containa  dictionary that maps image filenames ot annotations, such as
                                    produced by the methods in preprocessing.py (DC11 Utils)
        ann_format (string):        format of the annotations.
        box_dims (dict):            dimensions of the bounding boxes used for model training. For cases where 
                                    dimensions that differ from what is specified in the annotations were used. 
        vis_prob (str):             plotting probability. For each image, a random number between 0 and 1 is drawn, 
                                    and if it's below vis_prob, the image in question & the corresponding predictions
                                    will be visualized. 
        vis_density (int):          number of objects/animals that will trigger plotting. If the number of predictions 
                                    for an image is equal to or exceeds vis_prob, the image & predictions will be plotted
                                    regardless of vis_prob.
        patch_quality (float):      quality of tle image files (not sure what this does or why I need it tbh).
        save_pre_output (bool):     if True, outputs (plots + detections) before NMS will be sav ed as well.
        rm_tiles (bool):            whether to remove the tile image files after inference.
        save_patch_data (bool):     if True, the predictions for each tile will bes tored as files too (.json)
        verbose (bool):             if True, the amount of predictions removed by NMS will be printed to the console.
    Returns:
        None
    """

    assert patch_overlap < 1 and patch_overlap >= 0, \
        'Illegal tile overlap value {}'.format(patch_overlap)
    
    if save_patch_data: 
        raise NotImplementedError("This hasn't been implemented correctly yet!")
    
    # make directory for storing tiles
    tiling_dir = f"{imgs_dir}/tiles"
    Path(tiling_dir).mkdir()

    # make output folders
    det_dir = f"{output_dir}/detections"
    Path(det_dir).mkdir(exist_ok=True)

    if vis_prob > 0 or vis_density < math.inf: 
        vis_dir = f"{output_dir}/vis"
        Path(vis_dir).mkdir(exist_ok=True)
 
    model = YOLO(model_file)
   
    # read file if annotations are provided
    if ann_file:
        with open(ann_file, "r") as f:
            ann_dict = json.load(f)

    img_fns = list(Path(imgs_dir).rglob(f"*.{img_files_ext}"))
    # for accumulating total counts across images
    counts_sum = {cls_id: 0 for cls_id in class_ids}    
    
    print("*** Processing images")
    for fn in tqdm(img_fns, total=len(img_fns)):
        im = vis_utils.open_image(fn)
        patch_start_positions = get_patch_start_positions(img_width=im.width, img_height=im.height, patch_dims=patch_dims, 
                                                          overlap=patch_overlap) 
        
        patches = []
        # create tiles to perform inference on
        for patch in patch_start_positions: 
            patch_coords = {"x_min": patch[0], 
                            "y_min": patch[1], 
                            "x_max": patch[0] + patch_dims["width"] - 1,
                            "y_max": patch[1] + patch_dims["height"] - 1}
            
            patch_name = patch_info2name(image_name=fn.name, patch_x_min=patch_coords['x_min'], patch_y_min=patch_coords['y_min'])
            patch_fn = f"{tiling_dir}/{patch_name}.jpg"
            
            patch_metadata = {"patch_fn": patch_fn,
                              "patch_name": patch_name,
                              "coords": patch_coords}
            
            patches.append(patch_metadata)
        
            patch_im = im.crop((patch_coords["x_min"], patch_coords["y_min"], patch_coords["x_max"] + 1,
                                patch_coords["y_max"] + 1))
            assert patch_im.size[0] == patch_dims["width"]
            assert patch_im.size[1] == patch_dims["height"]
        
            assert not Path(patch_fn).exists()
            patch_im.save(patch_fn, quality=patch_quality)

    
        # create folder to store original patch level predictions
        if save_patch_data:
            patch_data_dir = Path(det_dir).parent / "patch_data"
            patch_data_dir.mkdir(parents=False, exist_ok=True)
            patch_output_dir = f"{patch_data_dir}/{fn.stem}"
            Path(patch_output_dir).mkdir(parents=False, exist_ok=True)
        else:
            patch_output_dir = None

        # run detection on patches 
        patch_fns = [patch["patch_fn"] for patch in patches]
        if task == "detect":
            predictions = model(patch_fns, iou=iou_thresh, verbose=False)
        else:
            predictions = model(patch_fns, radii=radii, dor=dor_thresh, verbose=False)
        
        # collect predictions from each patch and map it back to image level
        coords, conf, cls = collect_predictions_wrapper(task=task, class_ids=class_ids, predictions=predictions, patches=patches, 
                                                        device=model.device, patch_output_dir=patch_output_dir)
        
        # get counts before nms
        pre_nms = coords.shape[0]
        cls_idx_pre, counts_pre = torch.unique(cls.squeeze(1), return_counts=True)
        counts_pre_dict = {}
        for j in range(cls_idx_pre.shape[0]):
            counts_pre_dict[int(cls_idx_pre[j].item())] = int(counts_pre[j].item())

        if save_pre_output:
            with open(f"{det_dir}/{fn.stem}_pre_nms.json", "w") as f:
                json.dump(counts_pre_dict, f, indent=1)

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (pre_nms >= vis_density)
        if visualize and save_pre_output:
            plot_annotated_img(img_fn=str(fn), coords=coords, pre_nms=True, output_dir=vis_dir)

        # perform nms
        if task == "detect":
            idxs = torchvision.ops.nms(boxes=coords, scores=conf.squeeze(1), iou_threshold=iou_thresh)
        else:
            radii_t = generate_radii_t(radii=radii, cls=cls.squeeze(1))
            idxs = loc_nms(preds=coords, scores=conf.squeeze(1), radii=radii_t, dor_thres=dor_thresh)

        # combine coordinates, confidence and class into one tensor
        preds_img_final = torch.hstack((coords[idxs], conf[idxs], cls[idxs]))
        
        # If annotations are available, collect evaluation metrics at the image level 
        if ann_file:
            cfm_img = ConfusionMatrix(nc=len(class_ids), task=task)
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[fn.name], task=task, ann_format=ann_format, device=coords.device, 
                                            box_dims=box_dims)
            
            if task == "detect":
                cfm_img.process_batch(detections=preds_img_final, gt_bboxes=gt_coords, gt_cls=gt_cls)
            else:
                radii_t = generate_radii_t(radii=radii, cls=gt_cls)
                cfm_img.process_batch_loc(localizations=preds_img_final, gt_locs=gt_coords, gt_cls=gt_cls, radii=radii_t)

            # write confusion matrix to file
            with open(f"{det_dir}/{fn.stem}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)

        # get counts post nms 
        post_nms = preds_img_final.shape[0]
        cls_idx_post, counts_post = torch.unique(preds_img_final[:, -1], return_counts=True)
        counts_post_dict = {}
        for j in range(cls_idx_post.shape[0]):
            counts_post_dict[int(cls_idx_post[j].item())] = int(counts_post[j].item())

        with open(f"{det_dir}/{fn.stem}.json", "w") as f:
            json.dump(counts_post_dict, f, indent=1)
        
        if verbose:
            print(f"NMS removed {pre_nms - post_nms} predictions!")

        # add to count sum
        for class_idx, n in counts_post_dict.items():
            counts_sum[class_idx] += n

        if visualize:
            plot_annotated_img(img_fn=str(fn), coords=coords[idxs], cls=cls[idxs], pre_nms=False, output_dir=vis_dir)

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)

    if rm_tiles:
        shutil.rmtree(tiling_dir) 
