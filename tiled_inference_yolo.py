import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")
sys.path.append("/home/giacomo/projects/P0_YOLOcate/ultralytics/utils")

import os
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


def collect_boxes(predictions: list, patches: list, device: torch.device, patch_output_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_preds = torch.empty((0, 6), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.boxes

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordiantes, confidence and class into one tensor
        data_merged = torch.hstack((data.xyxy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # write patch level predictions to file
        patch_coords_str = f"xmin{patches[i]['coords']['x_min']}_ymin{patches[i]['coords']['y_min']}"
        np.savetxt(fname=f"{patch_output_dir}/{patch_coords_str}", X=data_merged.cpu().numpy())

        # map prediction to image_level
        data_merged[:, [0, 2]] = data_merged[:, [0,2]] + patches[i]["coords"]["x_min"]
        data_merged[:, [1, 3]] = data_merged[:, [1,3]] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((4, 1, 1), 1)


def collect_locations(predictions: list, patches: list, device: torch.device, patch_output_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_preds = torch.empty((0, 4), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.locations

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordiantes, confidence and class into one tensor
        data_merged = torch.hstack((data.xy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # write patch level predictions to file
        patch_coords_str = f"xmin{patches[i]['coords']['x_min']}_ymin{patches[i]['coords']['y_min']}"
        np.savetxt(fname=f"{patch_output_dir}/{patch_coords_str}", X=data_merged.cpu().numpy())

        # map prediction to image_level
        data_merged[:, 0] = data_merged[:, 0] - patches[i]["coords"]["x_min"]
        data_merged[:,1] = data_merged[:, 1] - patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((2, 1, 1), 1)


def collect_predictions_img_lvl(task: str,predictions: list, patches: list, device: torch.device, patch_output_dir: str) -> torch.Tensor:
    if task == "detect":
        return collect_boxes(predictions=predictions, patches=patches, device=device, patch_output_dir=patch_output_dir)
    else:
        return collect_locations(predictions=predictions, patches=patches, device=device, patch_output_dir=patch_output_dir)
    

def plot_annotated_img(img_fn: str, coords: torch.Tensor, pre_nms: bool, output_dir: str) -> None:

    img_arr = cv2.imread(img_fn)
    boxes = coords.shape[1] == 4

    for i in range(coords.shape[0]):
        if boxes:
            cv2.rectangle(img=img_arr, pt1=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                          pt2=(int(round(coords[i, 2].item())), int(round(coords[i, 3].item()))), color=(0, 0, 255), 
                          thickness=1)
        else:
            cv2.circle(img=img_arr, center=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                       radius=PT_VIS_CIRCLE_RADIUS, color=(0, 0, 255), thickness=-1)

    output_ext = "_pre_nms" if pre_nms else ""
            
    cv2.imwrite(f"{output_dir}/{Path(img_fn).stem}{output_ext}.jpg", img_arr)


def run_tiled_inference(model_file: str, task: str, radii: dict, class_ids: list, imgs_dir: str, img_files_ext: str, tiling_dir: str, 
                        patch_dims: dict, patch_overlap: float, patch_quality: int, meta_results_dir: str, vis_dir: str, det_dir: str, 
                        vis_prob: float, vis_pre: bool = False, iou_thresh: float = None, dor_thresh: float = None, max_offset: int = 1, 
                        rm_tiles: bool = True) -> None:
   
    """
    explain offset param with wh reference
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
        patch_output_dir = f"{meta_results_dir}/{fn.stem}"
        Path(patch_output_dir).mkdir(parents=False, exist_ok=True)

        # run detection on patches 
        patch_fns = [patch["patch_fn"] for patch in patches]
        predictions = model(patch_fns, verbose=False)
        
        # collect predictions from each patch mapped to image level
        coords, conf, cls = collect_predictions_img_lvl(task=task, predictions=predictions, patches=patches, 
                                                        device=model.device, patch_output_dir=patch_output_dir)
        
        visualize = random.randint(0, 1000) / 1000 <= vis_prob
        
        # get counts before nms
        pre_nms = coords.shape[0]
        cls_idx_pre, counts_pre = torch.unique(cls.squeeze(), return_counts=True)
        counts_pre_dict = {}
        for j in range(cls_idx_pre.shape[0]):
            counts_pre_dict[int(cls_idx_pre[j].item())] = int(counts_pre[j].item())

        with open(f"{det_dir}/{fn.stem}_pre_nms.json", "w") as f:
            json.dump(counts_pre_dict, f, indent=1)

        if (visualize and vis_pre) or fn.stem == "2017_Replicate_2017-10-01_Cam2_CAM25238":
            assert plot_annotated_img(img_fn=str(fn), coords=coords, pre_nms=True, output_dir=vis_dir), "plotting failed!"


        # add offset to separate ms per class
        c = cls * max_offset
        coords_shifted = coords + c

        # perform nms
        if task == "detect":
            idxs = torchvision.ops.nms(boxes=coords_shifted, scores=conf.squeeze(), iou_threshold=iou_thresh)
        else:
            radii_t = generate_radii_t(radii=radii, cls=cls.squeeze())
            idxs = loc_nms(preds=coords_shifted, scores=conf.squeeze(), radii=radii_t, dor_thresh=dor_thresh)

        preds_img_final = torch.hstack((coords[idxs], conf[idxs], cls[idxs]))

        # get counts post nms 
        post_nms = preds_img_final.shape[0]
        cls_idx_post, counts_post = torch.unique(preds_img_final[:, -1], return_counts=True)
        counts_post_dict = {}
        for j in range(cls_idx_post.shape[0]):
            counts_post_dict[int(cls_idx_post[j].item())] = int(counts_post[j].item())

        with open(f"{det_dir}/{fn.stem}_post_nms.json", "w") as f:
            json.dump(counts_post_dict, f, indent=1)
        
        if pre_nms != post_nms:
            print(f"NMS removed {pre_nms - post_nms} predictions!")

        # add to count sum
        for class_idx, n in counts_post_dict.items():
            counts_sum[class_idx] += n

        if visualize or fn.stem == "2017_Replicate_2017-10-01_Cam2_CAM25238":
            assert plot_annotated_img(img_fn=str(fn), coords=coords[idxs], pre_nms=False, output_dir=vis_dir), "plotting failed!"

    # save counts
    with open(f"{det_dir}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)

    if rm_tiles:
        os.rmdir(tiling_dir) 

            
