import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")
sys.path.append("/home/giacomo/projects/P0_YOLOcate/ultralytics/utils")

import shutil
import tqdm
import json
import random

import torch
import torchvision
import cv2
import visualization_utils as vis_utils
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from ops import loc_nms, generate_radii_t

from processing_utils import *

PT_VIS_RADIUS = 2


def collect_boxes(predictions: list, patches: list, device: torch.device, patch_output_dir: tuple[str, None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all patch-level boxes from a list of predictions and map them to the image space.
    Arguments:
        predictions (list):         list containing all patch predictions.
        patches (list):             list containing all patches.
        device (torch.device):      device used for inference.
        patch_output_dir (str):     path to the folder where patch-level predictions can be stored. 

    """

    all_preds = torch.empty((0, 6), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.boxes

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordiantes, confidence and class into one tensor
        data_merged = torch.hstack((data.xyxy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # write patch level predictions to file
        if patch_output_dir:
            patch_coords_str = f"xmin{patches[i]['coords']['x_min']}_ymin{patches[i]['coords']['y_min']}"
            np.savetxt(fname=f"{patch_output_dir}/{patch_coords_str}", X=data_merged.cpu().numpy())

        # map prediction to image_level
        data_merged[:, [0, 2]] = data_merged[:, [0,2]] + patches[i]["coords"]["x_min"]
        data_merged[:, [1, 3]] = data_merged[:, [1,3]] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((4, 1, 1), 1)


def collect_locations(predictions: list, patches: list, device: torch.device, patch_output_dir: tuple[str, None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all patch-level locations from a list of predictions and map them to the image space.
    Arguments:
        predictions (list):         list containing all patch predictions.
        patches (list):             list containing all patches.
        device (torch.device):      device used for inference.
        patch_output_dir (str):     path to the folder where patch-level predictions can be stored. 

    """

    all_preds = torch.empty((0, 4), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.locations

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordiantes, confidence and class into one tensor
        data_merged = torch.hstack((data.xy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # write patch level predictions to file
        if patch_output_dir:
            patch_coords_str = f"xmin{patches[i]['coords']['x_min']}_ymin{patches[i]['coords']['y_min']}"
            np.savetxt(fname=f"{patch_output_dir}/{patch_coords_str}", X=data_merged.cpu().numpy())

        # map prediction to image_level
        data_merged[:, 0] = data_merged[:, 0] + patches[i]["coords"]["x_min"]
        data_merged[:,1] = data_merged[:, 1] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image level tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((2, 1, 1), 1)


def collect_predictions_wrapper(task: str, predictions: list, patches: list, device: torch.device, patch_output_dir: tuple[str, None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper for 'collect_boxes' and 'collect_locations'.
    Arguments:
        task (str):                 string indicating the model task. Can be 'detect' or 'locate'.
        predictions (list):         list containing all predictions made on an image's patches.
        patches (list):             list containing all patches of an image.
        device (torch.device):      device used for inference.
        patch_output_dir (str):     path to the folder where patch-level predictions can be stored. 

    """
    if task == "detect":
        return collect_boxes(predictions=predictions, patches=patches, device=device, patch_output_dir=patch_output_dir)
    else:
        return collect_locations(predictions=predictions, patches=patches, device=device, patch_output_dir=patch_output_dir)
    

def plot_annotated_img(img_fn: str, coords: torch.Tensor, output_dir: str, pre_nms: bool = False,) -> None:
    """
    Plot an image and visualize the predictions it contains.
    Arguments:
        img_fn (str):               path to the imnage file
        coords (torch.Tensor):      tensor containing the prediction coordinates. Either 4 (boxes in xyxy format) or two (points) columns. 
        output_dir (str):           path to the folder where the plot can be stored.
        pre_nms (bool):             if True, the coords tensor is assumed to contain predictions before global NMS was applied.
    """

    img_arr = cv2.imread(img_fn)
    boxes = coords.shape[1] == 4

    for i in range(coords.shape[0]):
        if boxes:
            cv2.rectangle(img=img_arr, pt1=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                          pt2=(int(round(coords[i, 2].item())), int(round(coords[i, 3].item()))), color=(0, 255, 0), 
                          thickness=1)
        else:
            cv2.circle(img=img_arr, center=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                       radius=PT_VIS_RADIUS, color=(0, 255, 0), thickness=-1)

    output_ext = "_pre_nms" if pre_nms else ""
            
    assert cv2.imwrite(f"{output_dir}/{Path(img_fn).stem}{output_ext}.jpg", img_arr), "Plotting failed!"


def run_tiled_inference(model_file: str, task: str, class_ids: list, imgs_dir: str, img_files_ext: str, tiling_dir: str, 
                        patch_dims: dict, patch_overlap: float, vis_dir: str, det_dir: str, vis_prob: float, vis_density: int,
                        patch_quality: int = 95, radii: dict = None, save_pre_output: bool = False, iou_thresh: float = None, 
                        dor_thresh: float = None, max_offset: int = 1, rm_tiles: bool = True, save_patch_data: bool = False, 
                        verbose: bool = False) -> None:
   
    """
    Perform tiled inference on a directpry of images. 
    Arguments:
        model_file (str):           path to the model file (.pt) to be used for inference.
        task (str):                 task string. Can be 'detect' or 'locate'.
        class_ids (list):           list of integer class IDs in the data the model was trained on.
        imgs_dir (str):             path to the folder containing the images inference needs to be performed on.
        img_files_ext (str):        Extension of image files. E.g., 'JPG'.
        tiling_dir (str):           path to the folder where tiles can be stored temporarily.
        patch_dims (dict):          dictionary with keys 'width' & 'height' specifying the tile dimensions. 
        patch_overlap (float):      amount of overlap (fraction) between tiles. 
        vis_dir (str):              path to the folder where plots of images containing predictions can be stored.
        det_dir (str):              path to the folder where inference outputs can be stored (.json).
        vis_prob (str):             plotting probability. For each image, a random number between 0 and 1 is drawn, 
                                    and if it's below vis_prob, the image in question & the corresponding predictions
                                    will be visualized. 
        vis_density (int):          number of objects/animals that will trigger plotting. If the number of predictions 
                                    for an image is equal to or exceeds vis_prob, the image & predictions will be plotted
                                    regardless of vis_prob.
        patch_quality (float):      quality of tle image files (not sure what this does or why I need it tbh).
        radii (dict):               dictionary containing the radius values for localization NMS.
        save_pre_output (bool):     if True, outputs (plots + detections) before NMS will be sav ed as well.
        iou_thresh (float):         IoU threshold to use during NMS when stitching tiles together. 
        dor_thresh (float):         DoR threshold to use during NMS when stitching tiles together. 
        max_offset (int):           number by which the class of a prediction will be multiplied to obtain the 
                                    offset by which the predicted box/point coordinates will be shifted. If max_offset == 1, 
                                    no shifting will take place. It should be roughly the size of the largest possible bounding 
                                    box in pixel. 
        rm_tiles (bool):            whether to remove the tile image files after inference.
        save_patch_data (bool):     if True, the predictions for each tile will bes tored as files too (.json)
        verbose (bool):             if True, the amount of predictions removed by NMS will be printed to the console.
    Returns:
        None
    """

    assert patch_overlap < 1 and patch_overlap >= 0, \
        'Illegal tile overlap value {}'.format(patch_overlap)
    
    model = YOLO(model_file)
    
    img_fns = list(Path(imgs_dir).rglob(f"*.{img_files_ext}"))
    counts_sum = {cls_id: 0 for cls_id in class_ids}
    
    print("*** Processing images")
    for fn in tqdm(img_fns, total=len(img_fns)):
        im = vis_utils.open_image(fn)
                
        patch_start_positions = get_patch_start_positions(img_width=im.width, img_height=im.height, patch_dims=patch_dims, 
                                                          overlap=patch_overlap) 
        
        patches = []
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
            patch_data_dir = str(Path(det_dir).parent / "patch_data")
            patch_data_dir.mkdir(parents=False, exist_ok=True)
            patch_output_dir = f"{patch_data_dir}/{fn.stem}"
            Path(patch_output_dir).mkdir(parents=False, exist_ok=True)
        else:
            patch_output_dir = None

        # run detection on patches 
        patch_fns = [patch["patch_fn"] for patch in patches]
        if task == "detect":
            predictions = model(patch_fns, verbose=False)
        else:
            predictions = model(patch_fns, radii=radii, verbose=False)
        
        # collect predictions from each patch and map it back to image level
        coords, conf, cls = collect_predictions_wrapper(task=task, predictions=predictions, patches=patches, 
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

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or pre_nms >= vis_density

        if visualize and save_pre_output:
            plot_annotated_img(img_fn=str(fn), coords=coords, pre_nms=True, output_dir=vis_dir)


        # add offset to apply NMS separately to different classes
        c = cls * max_offset
        coords_shifted = coords + c

        # perform nms
        if task == "detect":
            idxs = torchvision.ops.nms(boxes=coords_shifted, scores=conf.squeeze(1), iou_threshold=iou_thresh)
        else:
            radii_t = generate_radii_t(radii=radii, cls=cls.squeeze(1))
            idxs = loc_nms(preds=coords_shifted, scores=conf.squeeze(1), radii=radii_t, dor_thres=dor_thresh)

        # combine coordinates, confidence and class into one tensor
        preds_img_final = torch.hstack((coords[idxs], conf[idxs], cls[idxs]))

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
            plot_annotated_img(img_fn=str(fn), coords=coords[idxs], pre_nms=False, output_dir=vis_dir)

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)

    if rm_tiles:
        shutil.rmtree(tiling_dir) 

            
