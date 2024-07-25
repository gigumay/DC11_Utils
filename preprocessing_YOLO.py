import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")

import os
import math
import cv2
import random 
import shutil

from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from globs import *
from processing_utils import *

import visualization_utils as visutils


def split_dataset(dataset_dir: str, img_format: str, val_frac: float, test_frac: float, rm_copies: bool = False, output_dir: str = None) -> tuple[dict, dict]:
    """
    Split a dataset into training-, validation, and test set. Creates a "train", "val" and "test" directory in the specified output 
    directory, containing the corresponding images. 
    Arguments:
        dataset_dir (str):          path to the directory containing the dataset to be split.
        img_format (str):           format of the image files. Will be used to select which files in the 
                                    data directory to select for splitting.
        val_frac (float):           fraction of the data to be used for validation.
        test_frac (float):          fraction of the data to be used for testing.
        rm_copies (bool):           whether to keep copies of the image files. 
        output_dir (str):           path to the directory where the splits will be stored. 
    Returns:
        Two dictionaries: one containign the paths to the "train", "val" and "test" directories, and one mapping 
        splits to image names. 
    """
    if not output_dir:
        output_dir = dataset_dir

    splits = {"train": [], "val": [], "test": []}

    img_fns = list(Path(dataset_dir).glob(f"*.{img_format}"))
   
    n_test_imgs = int(math.ceil(test_frac * len(img_fns)))
    n_val_imgs = int(math.ceil(val_frac * len(img_fns)))

    random.shuffle(img_fns)

    test_fns = img_fns[:n_test_imgs]
    val_fns = img_fns[n_test_imgs:(n_test_imgs + n_val_imgs)]
    train_fns = img_fns[n_test_imgs + n_val_imgs:] 
    
    test_dir = f"{output_dir}/test"
    val_dir = f"{output_dir}/val"
    train_dir = f"{output_dir}/train"

    Path(test_dir).mkdir()
    Path(val_dir).mkdir()
    Path(train_dir).mkdir()

    dirs = {"train": train_dir, "val": val_dir, "test": test_dir}

    for fn in test_fns:
        shutil.copy(fn, test_dir)
        splits["test"].append(f"{Path(fn).name}")
        if rm_copies:
            Path.unlink(fn)
    for fn in val_fns:
        shutil.copy(fn, val_dir)
        splits["val"].append(f"{Path(fn).name}")
        if rm_copies:
            Path.unlink(fn)
    for fn in train_fns:
        shutil.copy(fn, train_dir)
        splits["train"].append(f"{Path(fn).name}")
        if rm_copies:
            Path.unlink(fn)

    return splits, dirs


def get_boxes_in_patch_img_lvl(annotations_in_img: list, ann_format: str, patch_coords: dict, box_dims: dict = None) -> list:
    """
    For a given patch, create a list of dictionaries that for each annotation in the patch contain the
    bounding box center coordinates (at image level), the bounding box dimensions and the class of the annotation.
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
        ann_format (str):           format of the annotations. 
        patch_coords (dict):        dictionary specifying the coordinates (in pixel) of the patch in question
        box_dims (dict):            dictionary specifying the dimensions of bounding boxes. If None, 
                                    the box dimensions will be extracted from the annotations.
    Returns: 
        A list containing dictironaries with the coordinates (in pixel) of the box centers, the box dimensions, 
        and the corresponding class. 
    """
    patch_boxes = []
    bbox_key = DATA_ANN_FORMATS[ann_format]["label_key"]
    category_key = DATA_ANN_FORMATS[ann_format]["category_key"]
    xmin_idx = DATA_ANN_FORMATS[ann_format]["x_min_idx"]
    ymin_idx = DATA_ANN_FORMATS[ann_format]["y_min_idx"]
    width_idx = DATA_ANN_FORMATS[ann_format]["width_idx"]
    height_idx = DATA_ANN_FORMATS[ann_format]["height_idx"]


    for ann in annotations_in_img:                
        box_x_center = ann[bbox_key][xmin_idx] + (ann[bbox_key][width_idx]/2.0)
        box_y_center = ann[bbox_key][ymin_idx] + (ann[bbox_key][height_idx]/2.0)

        box_dims_checked = {}

        if box_dims:
            box_dims_checked["width"] = box_dims["width"]
            box_dims_checked["height"] = box_dims["height"]
        else: 
            box_dims_checked["width"] = ann[bbox_key][width_idx]
            box_dims_checked["height"] = ann[bbox_key][height_idx]
        
        box_x_min = box_x_center - (box_dims_checked["width"]/2.0)
        box_x_max = box_x_center + (box_dims_checked["width"]/2.0)
        box_y_min = box_y_center - (box_dims_checked["height"]/2.0)
        box_y_max = box_y_center + (box_dims_checked["height"]/2.0)
        
        patch_contains_box = patch_coords["x_min"] < box_x_max and box_x_min < patch_coords["x_max"] \
                             and patch_coords["y_min"] < box_y_max and box_y_min < patch_coords["y_max"]
        
        if patch_contains_box:
            patch_boxes.append({"x_center": box_x_center, "y_center": box_y_center, "box_dims": box_dims_checked,
                                "category_id": ann[category_key]})
    
    return patch_boxes

def get_points_in_patch_img_lvl(annotations_in_img: list, ann_format: str, patch_coords: dict) -> list:
    """
    For a given patch, create a list of dictionaries that for each annotation in the patch contain the
    point label coordinates (at image level) and the corresponding class.
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (points) in the image the patch was taken from.
        ann_format (str):           format of the annotations. 
        patch_coords (dict):        dictionary specifying the coordinates (in pixel) of the patch in question
    Returns: 
        A list containing dictionaries with the coordinates (in pixel) of point, and the corresponding class. 
    """
    patch_pts = []
    pt_key = DATA_ANN_FORMATS[ann_format]["label_key"]
    category_key = DATA_ANN_FORMATS[ann_format]["category_key"]
    x_idx = DATA_ANN_FORMATS[ann_format]["x_idx"]
    y_idx = DATA_ANN_FORMATS[ann_format]["y_idx"]

    for ann in annotations_in_img:                
        x = ann[pt_key][x_idx]
        y = ann[pt_key][y_idx]

        patch_contains_pt = patch_coords["x_min"] < x and x < patch_coords["x_max"] \
                            and patch_coords["y_min"] < y and y < patch_coords["y_max"]
        
        if patch_contains_pt:
            patch_pts.append({"x_center": x, "y_center": y, "category_id": ann[category_key]})
    
    return patch_pts



def get_boxes_at_patch_lvl(annotations_in_img: list, ann_format: str, patch_dims: dict, patch_coords: dict, 
                           categories: list, max_overhang: float, box_dims: dict = None) -> tuple[list, dict, int]:
    """
    For a given patch, create a list of the bounding boxes (as yolo-formatted dictionaries and with 
    patch-level coordinates) that lie within that patch. Boxes that span multiple patches will either be 
    clipped or discarded, depending on the overhang parameter.
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
        ann_format (str):           format of the annotations.
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories (list):          list of classes in the datset
        max_overhang (float):       max fraction of a bounding box that is allowed to exceed the 
                                    patch without being discarded
        box_dims (dict):            dictionary specifying the dimensions of bounding boxes. If None, 
                                    the box dimensions will be extracted from the annotations.
    Returns: 
        1.  a list containing YOLO-formatted bounding boxes that lie within the provided patch 
            (coordinates at patch-level). 
        2.  the class distribution in the patch
        3.  the number of boxes that were clipped 

    Note: for clipping boxes torch.clip would probably be the more readable solution
    """

    class_distr_patch = {cat: 0 for cat in categories}
    patch_boxes = get_boxes_in_patch_img_lvl(annotations_in_img=annotations_in_img, ann_format=ann_format,
                                             patch_coords=patch_coords, box_dims=box_dims)
    
    if not patch_boxes:
        return [], class_distr_patch, 0 
    
    n_clipped_boxes = 0
    yolo_boxes_patch = []

    for box_dict in patch_boxes:
        x_center_absolute_original = box_dict["x_center"]
        y_center_absolute_original = box_dict["y_center"]
        
        x_center_absolute_patch = x_center_absolute_original - patch_coords["x_min"]
        y_center_absolute_patch = y_center_absolute_original - patch_coords["y_min"]
        
        assert (1 + patch_coords["x_max"] - patch_coords["x_min"]) == patch_dims["width"]
        assert (1 + patch_coords["y_max"] - patch_coords["y_min"]) == patch_dims["height"]
        
        # Yolo expects relative coordinates
        x_center_relative = x_center_absolute_patch / patch_dims["width"]
        y_center_relative = y_center_absolute_patch / patch_dims["height"]
        
        '''
        calculate if and how far boxes exceed the patch, discard them if it's too much, 
        clip them otherwise. Remaining boxes are added to the list to be returned. 
        '''
        clipped_box = False
        # Yolo also expects relative box dimensions
        box_dims_rel = {"width": box_dict["box_dims"]["width"] / patch_dims["width"], 
                        "height": box_dict["box_dims"]["height"] / patch_dims["height"]}
        
        box_right = x_center_relative + (box_dims_rel["width"] / 2.0)                    
        if box_right > 1.0:
            overhang = box_right - 1.0
            # check the fraction of the box that exceeds the patch
            overhang_box_frac = overhang / box_dims_rel["width"]
            if overhang_box_frac > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["width"] -= overhang
            x_center_relative -= overhang / 2.0

        box_bottom = y_center_relative + (box_dims_rel["height"] / 2.0)                                        
        if box_bottom > 1.0:
            overhang = box_bottom - 1.0
            # check the fraction of the box that exceeds the patch
            overhang_box_frac = overhang / box_dims_rel["height"]
            if overhang_box_frac > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["height"] -= overhang
            y_center_relative -= overhang / 2.0
        
        box_left = x_center_relative - (box_dims_rel["width"] / 2.0)
        if box_left < 0.0:
            overhang = abs(box_left)
            # check the fraction of the box that exceeds the patch
            overhang_box_frac = overhang / box_dims_rel["width"]
            if overhang_box_frac > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["width"] -= overhang
            x_center_relative += overhang / 2.0
            
        box_top = y_center_relative - (box_dims_rel["height"] / 2.0)
        if box_top < 0.0:
            overhang = abs(box_top)
            # check the fraction of the box that exceeds the patch
            overhang_box_frac = overhang / box_dims_rel["height"]
            if overhang_box_frac > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["height"] -= overhang
            y_center_relative += overhang / 2.0
            
        if clipped_box:
            n_clipped_boxes += 1
        
        # YOLO annotations are category, x_center, y_center, w, h
        yolo_box = [box_dict["category_id"], x_center_relative, y_center_relative, 
                    box_dims_rel["width"], box_dims_rel["height"]]
        
        yolo_boxes_patch.append(yolo_box)
        class_distr_patch[box_dict["category_id"]] += 1

    return yolo_boxes_patch, class_distr_patch, n_clipped_boxes



def get_points_at_patch_lvl(annotations_in_img: list, ann_format: str, is_bxs: bool, patch_dims: dict, patch_coords: dict, categories: list,
                            box_dims: dict = None) -> tuple[list, dict]:
    """
    For a given patch, create a list of point labels (as yolo-formatted dictionaries) that lie within that patch
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (point labels) in the image the patch
                                    was taken from.
        ann_format (str):           format of the annotations.
        is_bxs (bool):              indicates whether the annotations come in the form of boxes or points
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories (list):          list of classes in the datset
        box_dims (dict):            bounding box dimensions for when box inputs are used
    Returns: 
        1.  a list containing  dictionaries with the coordinates (in pixel and at patch level) of the point labels that
            lie within the provided patch, as well as the corresponding class.
        2.  the class distribution in the patch
    """
    class_distr_patch = {cat: 0 for cat in categories}
    gt_points = []
            
    assert (1 + patch_coords["x_max"] - patch_coords["x_min"]) == patch_dims["width"]
    assert (1 + patch_coords["y_max"] - patch_coords["y_min"]) == patch_dims["height"]

    if is_bxs: 
        ann_list = get_boxes_in_patch_img_lvl(annotations_in_img=annotations_in_img, ann_format=ann_format, patch_coords=patch_coords,
                                              box_dims=box_dims)
    else:
        ann_list = get_points_in_patch_img_lvl(annotations_in_img=annotations_in_img, ann_format=ann_format, patch_coords=patch_coords)
    
    for ann in ann_list: 
        gt_x_img = ann["x_center"]
        gt_y_img = ann["y_center"]
            

        patch_contains_pt = (patch_coords["x_min"] < gt_x_img and gt_x_img < patch_coords["x_max"] \
                             and patch_coords["y_min"] < gt_y_img and gt_y_img < patch_coords["y_max"])
        
        if patch_contains_pt:
            gt_x_patch = gt_x_img - patch_coords["x_min"]
            gt_y_patch = gt_y_img - patch_coords["y_min"]

            #Again, relative coordinates 
            x_coords_relative_patch = gt_x_patch / patch_dims["width"]
            y_coords_relative_patch = gt_y_patch / patch_dims["height"]
            
            gt_points.append([ann["category_id"], x_coords_relative_patch, y_coords_relative_patch])
            class_distr_patch[ann["category_id"]] += 1
    
    return gt_points, class_distr_patch



def get_annotations_in_patch(annotations_in_img: list, ann_format: str, boxes_in: bool, boxes_out: bool, patch_dims: dict, patch_coords: dict, 
                             categories: list, box_dims: dict, max_box_overhang: float = 0.85)-> tuple[list, dict, int]:
    """
    Retrieves the annotations in a given patch. 
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes) in the image 
                                    the patch was taken from.
        ann_format (str):           format of the annotations. 
        boxes_in (bbol):            must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):           if false, return point labels instead of bounding boxes. Must be set to false
                                    when working iwht point labels. 
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories (list):          list of classes in the datset
        box_dims (dict):            dictionary specifying the dimensions of bounding boxes. If None, 
                                    the box dimensions will be extracted from the annotations.
        max_box_overhang (float):   maximum percentage of a bounding box that can lie outside of a patch without causing 
                                    the box to be excluded.
    Returns:
        1. list containing the annotations within the patch (points or boxes)
        2. dictionary containing the distribution of classes in the patch 
        3. the number of boxes that had to be clipped        
    """

    assert not (not boxes_in and boxes_out)
    if boxes_in: 
        yolo_boxes, class_distr, n_clipped_boxes = get_boxes_at_patch_lvl(annotations_in_img=annotations_in_img,
                                                                          ann_format=ann_format,
                                                                          patch_dims=patch_dims, 
                                                                          patch_coords=patch_coords, 
                                                                          categories=categories,
                                                                          max_overhang=max_box_overhang,
                                                                          box_dims=box_dims)
        if boxes_out:
            return yolo_boxes, class_distr, n_clipped_boxes
        else:
            gt_points, class_distr = get_points_at_patch_lvl(annotations_in_img=annotations_in_img, 
                                                             ann_format=ann_format, is_bxs=True, 
                                                             patch_dims=patch_dims, patch_coords=patch_coords,
                                                             categories=categories, box_dims=box_dims)
            return gt_points, class_distr, 0       
    else: 
         gt_points, class_distr = get_points_at_patch_lvl(annotations_in_img=annotations_in_img, 
                                                          ann_format=ann_format, is_bxs=False,
                                                          patch_dims=patch_dims, patch_coords=patch_coords, 
                                                          categories=categories)
         return gt_points, class_distr, 0
        
    

def process_image(img_path: str, img2ann: dict, classes: list, data_ann_format: str, model_ann_format: str, boxes_in: bool, boxes_out: bool, 
                  patch_dims: dict, patch_overlap: float, patch_jpeg_quality: int,  output_dir: str, box_dims: dict = None, radii: dict = None, 
                  visualize: bool = False, vis_output_dir: str = None) -> dict:
    """
    Process a given image. Processing consists of dividing the image into patches and assigning each 
    patch a set of annotations (boxes or points) that lie within that patch.
    Arguments: 
        img_path  (str):                path to the directory where the image is stored
        img2ann (dict):                 dictionary mapping image names (= file name without file extension)
        classes (list):                 list containing class ids
        data_ann_format (str):          string specifying the format of the ground truth annotations
        model_ann_fomrat (str):         annotation format expected by the downstream model.
        boxes_in (bbol):                must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):               if false, return point labels instead of bounding boxes. Must be set to false
                                        when working with point labels. 
        patch_dims (dict):              dict specifying the dimensions of the patches
        patch_overlap (float):          overlap between patches    
        patch_jpeg_quality (int):       quality of the patch-images  
        output_dir (str):               path to the directory where the patch images and annotations will be stored.
        box_dims (dict):                dictionary specifying the dimensions of bounding boxes. If not set,  box dimensions
                                        will be extracted from the annotations
        radii (dict):                   dictionary containing the radii for each class         
        visualize (bool):               if true, the annotations are drawn into the image and the patches,
                                        and the visualizations are then stored in 'vis_output_dir'       
        vis_output_dir (str):           path to the directory where the visualizations will be stored
    Returns: 
        a dictionary containing:    1.  a dictionary mapping patch names to patch metadata (coordinates, annotations, etc.)
                                        for non-empty patches
                                    2.  a dictionary mapping patch names to patch metadata (coordinates, annotations, etc.)
                                        for empty patches
                                    3.  the total number of patches extracted from the image 
                                    4.  the number of non-empty patches extracted from the image
                                    5.  the number of annotations found in the image
                                    6.  the number of boxes that had to be clipped during the patching.
                                    7.  the class distribution in the image
    """

    pil_img = visutils.open_image(img_path)
    img_name = Path(img_path).stem

    if radii: 
        # make sure there is a radius for every class
        assert len(classes) == len(list(radii.keys())) and all(cat in radii.keys() for cat in classes), \
        f"Found classes {classes}, but radii only for {list(radii.keys())}"

    annotations = img2ann[img_name]
    pos_patch_metadata_mapping_img = {}
    neg_patch_metadata_mapping_img = {}
    n_annotations_img = 0
    n_boxes_clipped_img = 0
    class_distr_img = {cat: 0 for cat in classes}

    patch_start_positions = get_patch_start_positions(img_width=pil_img.size[0], img_height=pil_img.size[1], patch_dims=patch_dims, overlap=patch_overlap)
 

    for patch in patch_start_positions:
        patch_coords = {"x_min": patch[PATCH_XSTART_IDX], 
                        "y_min": patch[PATCH_YSTART_IDX], 
                        "x_max": patch[PATCH_XSTART_IDX] + patch_dims["width"] - 1,
                        "y_max": patch[PATCH_YSTART_IDX] + patch_dims["height"] - 1}
        

        gt, patch_distr, n_boxes_clipped = get_annotations_in_patch(annotations_in_img=annotations, 
                                                                    ann_format=data_ann_format,
                                                                    boxes_in=boxes_in, 
                                                                    boxes_out=boxes_out, 
                                                                    patch_dims=patch_dims, 
                                                                    patch_coords=patch_coords, 
                                                                    categories=classes, 
                                                                    box_dims=box_dims)
        
        if not gt:
            patch_name = patch_info2name(img_name, patch_coords["x_min"], patch_coords["y_min"],
                                             is_empty=True)
            patch_ann_file = Path(output_dir) / f"{patch_name}.txt"
                
            patch_metadata = {
                "patch_name": patch_name,
                "original_image_name": img_name,
                "patch_x_min": patch_coords["x_min"],
                "patch_y_min": patch_coords["y_min"],
                "patch_x_max": patch_coords["x_max"],
                "patch_y_max": patch_coords["y_max"], 
                "boxes": None,
                "points": None, 
                "class_distribution": None
            }
        
            neg_patch_metadata_mapping_img[patch_name] = patch_metadata
        else:
            n_boxes_clipped_img += n_boxes_clipped
            n_annotations_img += len(gt)
            for class_id in class_distr_img.keys():
                class_distr_img[class_id] += patch_distr[class_id]

            patch_name = patch_info2name(img_name, patch_coords["x_min"], patch_coords["y_min"],
                                         is_empty=False)
            patch_ann_file = Path(output_dir) / f"{patch_name}.txt"
            

            patch_metadata = {
                "patch_name": patch_name,
                "original_image_name": img_name,
                "patch_x_min": patch_coords["x_min"],
                "patch_y_min": patch_coords["y_min"],
                "patch_x_max": patch_coords["x_max"],
                "patch_y_max": patch_coords["y_max"],
                "boxes": gt if boxes_out else None,
                "points": gt if not boxes_out else None,
                "class_distribution": patch_distr
            }
        
            pos_patch_metadata_mapping_img[patch_name] = patch_metadata
        
        
        assert not patch_ann_file.exists()
        
    
        '''
        PIL represents coordinates in a way that is very hard for me to get my head
        around, such that even though the "right" and "bottom" arguments to the crop()
        function are inclusive... well, they're not really.
        
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
        
        So we add 1 to the max values.
        '''
        patch_im = pil_img.crop((patch_coords["x_min"], patch_coords["y_min"], patch_coords["x_max"] + 1,
                                patch_coords["y_max"] + 1))
        assert patch_im.size[0] == patch_dims["width"]
        assert patch_im.size[1] == patch_dims["height"]
        

        patch_image_file = Path(output_dir) / f"{patch_name}.jpg"
        assert not patch_image_file.exists()
        patch_im.save(patch_image_file, quality=patch_jpeg_quality)
        
        with open(patch_ann_file, 'w') as f:
            for ann in gt:
                if boxes_out:
                    ann_str = f"{ann[MODEL_ANN_FORMATS[model_ann_format]['category_idx']]} " \
                              f"{ann[MODEL_ANN_FORMATS[model_ann_format]['center_x_idx']]} " \
                              f"{ann[MODEL_ANN_FORMATS[model_ann_format]['center_y_idx']]} " \
                              f"{ann[MODEL_ANN_FORMATS[model_ann_format]['width_idx']]} " \
                              f"{ann[MODEL_ANN_FORMATS[model_ann_format]['height_idx']]}\n"
                else:
                    ann_str = f"{ann[MODEL_ANN_FORMATS[model_ann_format]['category_idx']]} " \
                              f"{radii[MODEL_ANN_FORMATS[model_ann_format]['category_idx']]} " \
                              f"{ann[MODEL_ANN_FORMATS[model_ann_format]['x_idx']]} " \
                              f"{ann[MODEL_ANN_FORMATS[model_ann_format]['y_idx']]}\n"
                
                f.write(ann_str)
                
        assert Path.exists(patch_ann_file)

        if visualize and gt:
            vis_processed_img(img_path=img_path, img2ann=img2ann, data_ann_format=data_ann_format, model_ann_format=model_ann_format, 
                              patch_metadata_mapping_img=pos_patch_metadata_mapping_img, patch_dims=patch_dims, boxes_in=boxes_in, 
                              boxes_out=boxes_out, output_dir=vis_output_dir)

    # sanity check
    assert len(patch_start_positions) == (len(pos_patch_metadata_mapping_img) + len(neg_patch_metadata_mapping_img))
    
    return {"pos_patches_mapping": pos_patch_metadata_mapping_img, 
            "neg_patches_mapping": neg_patch_metadata_mapping_img,
            "n_total_patches": len(patch_start_positions),
            "n_non_empty_patches":len(pos_patch_metadata_mapping_img),
            "n_empty_patches": len(neg_patch_metadata_mapping_img),
            "n_annotations": n_annotations_img, 
            "n_boxes_clipped":  n_boxes_clipped_img, 
            "class_distribution": class_distr_img}



def vis_processed_img(img_path, img2ann: dict, data_ann_format: str, model_ann_format: str, patch_metadata_mapping_img: dict, patch_dims: dict, 
                      boxes_in: bool, boxes_out: bool, output_dir: str, circle_radius: int = 3) -> None:
    """
    Draws annotations into the provided image and the corresponding patches. Image and patches are then stored in the specified 
    directory.
    Arguments 
        img_path  (str):                path to the directory where the image is stored
        img2ann (dict):                 dictionary mapping image names (= file name without file extension)
        data_ann_format (str):          string specifying the format of the ground truth annotations
        model_ann_fomrat (str):         annotation format expected by the downstream model.
        patch_metadata_mapping_img (dict):  dictionary containing the metadata for all patches of the image in question
        patch_dims (dict):                  dictionary containing the dimensions of patches
        boxes_in (bbol):                    must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):                   if false, the annotations are expected to contain point labels instead of bounding
                                            boxes
        output_dir (str):                   directory where the output files will be stored
        circle_radius (int):                raidus of the circles drawn for point label visualization.
    Returns:
        None
    """
    img_name = Path(img_path).stem

    output_path = Path(output_dir) / img_name
    output_path.mkdir(parents=False, exist_ok=True)

    img_arr = cv2.imread(img_path)

    # plot entire image with annotations
    for ann in img2ann[img_name]:
        label = ann[DATA_ANN_FORMATS[data_ann_format]["label_key"]]
        
        if boxes_out:
            #cv2 needs int coordinates
            xmin_img = round(label[DATA_ANN_FORMATS[data_ann_format]["x_min_idx"]])
            xmax_img = round(label[DATA_ANN_FORMATS[data_ann_format]["x_min_idx"]] + label[DATA_ANN_FORMATS[data_ann_format]["width_idx"]])
            ymin_img = round(label[DATA_ANN_FORMATS[data_ann_format]["y_min_idx"]])
            ymax_img = round(label[DATA_ANN_FORMATS[data_ann_format]["y_min_idx"]] + label[DATA_ANN_FORMATS[data_ann_format]["height_idx"]])
         
            cv2.rectangle(img=img_arr, pt1=(xmin_img, ymin_img), pt2=(xmax_img, ymax_img), 
                          color=(0, 0, 255), thickness=1)
        else:
            if boxes_in:
                x_center = round(label[DATA_ANN_FORMATS[data_ann_format]["x_min_idx"]] +label[DATA_ANN_FORMATS[data_ann_format]["width_idx"]] / 2.0)
                y_center = round(label[DATA_ANN_FORMATS[data_ann_format]["y_min_idx"]] + label[DATA_ANN_FORMATS[data_ann_format]["height_idx"]] / 2.0)
            else:
                x_center = label[DATA_ANN_FORMATS[data_ann_format]["x_idx"]]
                y_center = label[DATA_ANN_FORMATS[data_ann_format]["y_idx"]]
            
            cv2.circle(img=img_arr, center=(x_center, y_center), radius=circle_radius, 
                       color=(0, 0, 255), thickness=-1)

    cv2.imwrite(str(output_path/f"full_img_{img_name}.jpg"), img_arr)


    # draw annotations into patches
    for key in patch_metadata_mapping_img:
        patch_dict = patch_metadata_mapping_img[key]
        patch_arr = img_arr[patch_dict["patch_y_min"] : patch_dict["patch_y_max"] + 1, patch_dict["patch_x_min"] : 
                            patch_dict["patch_x_max"] + 1]

        if boxes_out:
            gt = patch_dict["boxes"]
        else:
            gt = patch_dict["points"]


        for ann in gt:
            if boxes_out:
                # convert relative (yolo-formatted) box data to absolute values for plotting
                x_center_absolute_patch = ann[MODEL_ANN_FORMATS[model_ann_format]["center_x_idx"]] * patch_dims["width"]
                y_center_absolute_patch = ann[MODEL_ANN_FORMATS[model_ann_format]["center_y_idx"]] * patch_dims["height"]
                width_absolute = ann[MODEL_ANN_FORMATS[model_ann_format]["width_idx"]] * patch_dims["width"]
                height_absolute = ann[MODEL_ANN_FORMATS[model_ann_format]["height_idx"]] * patch_dims["height"]

                xmin_patch = round(x_center_absolute_patch - (width_absolute / 2.0))
                xmax_patch = round(x_center_absolute_patch + (width_absolute / 2.0))
                ymin_patch = round(y_center_absolute_patch - (height_absolute / 2.0))
                ymax_patch = round(y_center_absolute_patch + (height_absolute / 2.0))

                cv2.rectangle(img=patch_arr, pt1=(xmin_patch, ymin_patch), pt2=(xmax_patch, ymax_patch), 
                              color=(0, 255, 0), thickness=1)
            else:
                # convert relative coords to pixel for plotting
                x_patch = round(ann[MODEL_ANN_FORMATS[model_ann_format]["x_idx"]] * patch_dims["width"])
                y_patch = round(ann[MODEL_ANN_FORMATS[model_ann_format]["y_idx"]] * patch_dims["height"])

                if x_patch == patch_dims["width"]:
                    x_patch -= 1
                if y_patch == patch_dims["height"]:
                    y_patch -= 1

                #check if circle fits into patch
                draw_circle = x_patch + circle_radius < patch_dims["width"] and \
                              x_patch - circle_radius > 0 and \
                              y_patch + circle_radius < patch_dims["height"] and \
                              y_patch - circle_radius > 0

                if draw_circle: 
                    cv2.circle(img=patch_arr, center=(x_patch, y_patch), radius=circle_radius, 
                               color=(0, 255, 0), thickness=-1)
                else:
                    patch_arr[y_patch, x_patch] = (0, 255, 0)

            cv2.imwrite(str(output_path / f"{patch_dict['patch_name']}.jpg"), patch_arr)



def patchify_imgs(img_paths: list, img2ann: dict, classes: list, data_ann_format: str, model_ann_format: str, boxes_in: bool, boxes_out: bool, patch_dims: dict, 
                  patch_overlap: float, neg_frac: float, output_dir: str, box_dims: dict = None, radii: dict = None, patch_jpeg_quality: int = 95,
                  rm_orig_imgs: bool = False, n_debug_imgs: int = -1, vis_prob: float = 0.0, vis_output_dir: str = None) -> dict:
    
    """
    Patchify a set of images. The amount of empty patches to be kept can be set via the 'neg_frac' parameter. 
    Arguments:
        img_path  (str):                path to the directory where the image is stored
        img2ann (dict):                 dictionary mapping image names (= file name without file extension)
        classes (list):                 list containing class ids
        data_ann_format (str):          string specifying the format of the ground truth annotations
        model_ann_fomrat (str):         annotation format expected by the downstream model.
        boxes_in (bbol):                must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):               if false, return point labels instead of bounding boxes. Must be set to false
                                        when working with point labels. 
        patch_dims (dict):              dict specifying the dimensions of the patches
        patch_overlap (float):          overlap between patches    
        neg_frac (float):               fraction of empty patches to keep
        output_dir (str):               path to the directory where the patch images and annotations will be stored.
        box_dims (dict):                dictionary specifying the dimensions of bounding boxes. If not set,  box dimensions
                                        will be extracted from the annotations
        radii (dict):                   dictionary containing the radii for each class  
        patch_jpeg_quality (int):       quality of the patch-images  
        rm_orig_imgs (bool):            if true, the full images are deleted after patchifying
        n_debug_imgs (int):             amount of images to process. If set to -1, all images are processed
        vis_prob (float):               probability that a processed image is visualized (i.e., plotted with the corresponding
                                        annotations)
        vis_output_dir (str):           path to the directory where plotted images can be stored. 
    Return: 
        A dictionary indicating the class distribution in the data. 
    

    """
    
    all_pos_patches_mapping = {}
    all_neg_patches_mapping = {}
    overall_distr = {cls_id: 0 for cls_id in classes}
    n_anns = 0
    n_boxes_clipped_total = 0
    n_patches_with_ann= 0
    n_patches_with_ann = 0
    n_patches_total = 0
    
    for i_img, p in tqdm(enumerate(img_paths), total=len(img_paths)):
        if n_debug_imgs > 0 and i_img > n_debug_imgs:
            break

        # visualize some outputs at random
        visualize = ((random.randint(0, 1000) / 1000)) <= vis_prob

        output = process_image(img_path=p,
                               img2ann=img2ann, 
                               classes=classes, 
                               data_ann_format=data_ann_format, 
                               model_ann_format=model_ann_format,  
                               boxes_in=boxes_in, 
                               boxes_out=boxes_out, 
                               patch_dims=patch_dims,
                               patch_overlap=patch_overlap, 
                               patch_jpeg_quality=patch_jpeg_quality,
                               visualize=visualize,
                               output_dir=output_dir,
                               box_dims=box_dims, 
                               radii=radii, 
                               vis_output_dir=vis_output_dir)
        
        
        
        # prevent overwriting (sanity check)
        for key in output["pos_patches_mapping"]:
            assert key not in all_pos_patches_mapping
        for key in output["neg_patches_mapping"]:
            assert key not in all_neg_patches_mapping

        all_pos_patches_mapping.update(output["pos_patches_mapping"])
        all_neg_patches_mapping.update(output["neg_patches_mapping"])

        n_patches_with_ann += output["n_non_empty_patches"]
        n_anns += output["n_annotations"]
        n_boxes_clipped_total += output["n_boxes_clipped"]
        n_patches_total += output["n_total_patches"]

        for cls_id in overall_distr.keys():
            overall_distr[cls_id] += output["class_distribution"][cls_id]

        if rm_orig_imgs:
            Path.unlink(Path(p))


    neg_patches = [p for p in Path(output_dir).glob("*.jpg") if "empty" in str(p)]

    #sanity check
    assert len(neg_patches) == len(all_neg_patches_mapping)

    # get number of empty patches to sample
    n_patches_all = int(math.ceil(len(all_pos_patches_mapping) / (1 - neg_frac)))
    n_negs = n_patches_all - len(all_pos_patches_mapping)

    random.shuffle(neg_patches)
    negs2rm = neg_patches[n_negs:]

    for neg_path in negs2rm:
        Path.unlink(neg_path)
        Path.unlink(neg_path.with_suffix(".txt"))
    


    print(f"\n\n\nTotal number of images/patches processed: {len(img2ann)}/{n_patches_total}\n"
          f"Processed {n_anns} points.\n"
          f"Out of {n_patches_total} patches, {n_patches_with_ann} patches containing valid annotations were obtained.\n" \
          f"{n_negs} empty patches were kept.\n")
    
    return overall_distr
