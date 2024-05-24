import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")

import math
import cv2
import random 
import json
import shutil

from pathlib import Path
from tqdm import tqdm

from pathlib import Path

import visualization_utils as visutils



def get_boxes_in_patch_img_lvl(annotations_in_img: list, patch_coords: dict, box_dims: dict = None) -> list:
    """
    For a given patch, create a list of dictionaries that for each annotation in the patch contain the
    bounding box center coordinates (at image level), the bounding box dimensions and the class of the annotation.
    Annotations are expected to be in the COCO-format ([xmin, ymin, width, height]).
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
        patch_coords (dict):        dictionary specifying the coordinates (in pixel) of the 
                                    patch in question
        box_dims (dict):            dictionary specifying the dimensions of bounding boxes. If None, 
                                    the box dimensions will be extracted from the annotations.
    Returns: 
        A list containing dictironaries with the coordinates (in pixel) of the box centers, the box dimensions, 
        and the corresponding class. 
    """
    patch_boxes = []
    for ann in annotations_in_img:                
        # In the input annotations, boxes are expected as x/y/w/h
        box_x_center = ann['bbox'][COCO_BOX_XMIN_IDX] + (ann['bbox'][COCO_BOX_WIDTH_IDX]/2.0)
        box_y_center = ann['bbox'][COCO_BOX_YMIN_IDX] + (ann['bbox'][COCO_BOX_HEIGHT_IDX]/2.0)

        box_dims_checked = {}

        if box_dims:
            box_dims_checked["width"] = box_dims["width"]
            box_dims_checked["height"] = box_dims["height"]
        else: 
            box_dims_checked["width"] = ann['bbox'][COCO_BOX_WIDTH_IDX]
            box_dims_checked["height"] = ann['bbox'][COCO_BOX_HEIGHT_IDX]
        
        box_x_min = box_x_center - (box_dims_checked["width"]/2.0)
        box_x_max = box_x_center + (box_dims_checked["width"]/2.0)
        box_y_min = box_y_center - (box_dims_checked["height"]/2.0)
        box_y_max = box_y_center + (box_dims_checked["height"]/2.0)
        
        patch_contains_box = patch_coords["x_min"] < box_x_max and box_x_min < patch_coords["x_max"] \
                             and patch_coords["y_min"] < box_y_max and box_y_min < patch_coords["y_max"]
        
        if patch_contains_box:
            patch_boxes.append({"x_center": box_x_center, "y_center": box_y_center, "box_dims": box_dims_checked,
                                "category_id": ann['category_id']})
    
    return patch_boxes



def get_boxes_at_patch_lvl(annotations_in_img: list, patch_dims: dict, patch_coords: dict, categories: list, 
                           max_overhang: float, box_dims: dict = None) -> tuple[list, dict, int]:
    """
    For a given patch, create a list of the bounding boxes (as yolo-formatted dictionaries and with 
    patch-level coordinates) that lie within that patch. Boxes that span multiple patches will either be 
    clipped or discarded, depending on the overhang parameter.
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
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
    patch_boxes = get_boxes_in_patch_img_lvl(annotations_in_img=annotations_in_img, 
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



def get_points_in_patch(annotations_in_img: list, is_bxs: bool, patch_dims: dict, patch_coords: dict, categories: list,
                        box_dims: dict = None) -> tuple[list, dict]:
    """
    For a given patch, create a list of point labels (as yolo-formatted dictionaries) that lie within that patch
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (point labels) in the image the patch
                                    was taken from.
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
        ann_list = get_boxes_in_patch_img_lvl(annotations_in_img=annotations_in_img, patch_coords=patch_coords,
                                              box_dims=box_dims)
    else:
        raise(NotImplementedError, "Loading point labels has not been implemented yet!")
    
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



def get_annotations_in_patch(annotations_in_img: list, boxes_in: bool, boxes_out: bool, patch_dims: dict, 
                             patch_coords: dict, categories: list, box_dims: dict)-> tuple[list, dict, int]:
    """
    Retrieves the annotations in a given patch. 
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes) in the image 
                                    the patch was taken from.
        boxes_in (bbol):            must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):           if false, return point labels instead of bounding boxes. Must be set to false
                                    when working iwht point labels. 
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories (list):          list of classes in the datset
        box_dims (dict):            dictionary specifying the dimensions of bounding boxes. If None, 
                                    the box dimensions will be extracted from the annotations.
    Returns:
        1. list containing the annotations within the patch (points or boxes)
        2. dictionary containing the distribution of classes in the patch 
        3. the number of boxes that had to be clipped        
    """

    # sanity check

    assert not (not boxes_in and boxes_out)
    if boxes_in: 
        yolo_boxes, class_distr, n_clipped_boxes = get_boxes_at_patch_lvl(annotations_in_img=annotations_in_img,
                                                                          patch_dims=patch_dims,patch_coords=patch_coords, 
                                                                          categories=categories,
                                                                          max_overhang=MAX_OVERHANG_BXS,box_dims=box_dims)
        if boxes_out:
            return yolo_boxes, class_distr, n_clipped_boxes
        else:
            gt_points, class_distr = get_points_in_patch(annotations_in_img=annotations_in_img, is_bxs=True, 
                                                         patch_dims=patch_dims, patch_coords=patch_coords,
                                                         categories=categories, box_dims=box_dims)
            return gt_points, class_distr, 0       
    else: 
         gt_points, class_distr = get_points_in_patch(annotations_in_img=annotations_in_img, is_bxs=False,
                                                      patch_dims=patch_dims, patch_coords=patch_coords, 
                                                      categories=categories)
         return gt_points, class_distr, 0
        
    


def process_image(source_dir_img: str, img: dict, img_width: int, img_height: int, boxes_in: bool, boxes_out: bool, 
                  img_id_to_ann: dict, patch_dims: dict, patch_start_positions: list, n_empties: int, 
                  patch_jpeg_quality: int, categories: list, visualize: bool, files_output_dir: str, 
                  box_dims: dict = None, radii: dict = None, vis_output_dir: str = None) -> dict:
    """
    Process a given image. Processing consists of dividing the image into patches and assigning each 
    patch a set of annotations (boxes or points) that lie within that patch. If the corresponding parameters are
    set accordingly the patches are saved as image files and box-/point-metadata is written to files in yolo-format 
    (happens always for patches that contain bounding boxes, but if desired empty files can be written for empty patches
    as well). 
    Arguments: 
        source_dir_img (str):           path to the directory where the image is stored
        img (dict):                     dictionary containing the image metadata
        img_width (int):                width of the image
        img_height (int):               height of the image
        boxes_in (bbol):                must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):               if false, return point labels instead of bounding boxes. Must be set to false
                                        when working iwht point labels. 
        img_id_to_ann (dict):           a dictionary mapping image ids to lists of annotations contained
                                        in the respective images. bounding boxes are expected in the COCO-format.
        patch_dims (dict):              dict specifying the dimensions of the patches.
        patch_start_positions (list):   list of pixel coordinates specifying the starting positions of the 
                                        patches
        n_empties (int):                the number of empty patches to sample
        patch_jpeg_quality (int):       quality of the patch-images
        categories:                     list of classes in the datset
        visualize (bool):               if true, the annotations are drawn into the image and the patches,
                                        which are then written into a specified directory.
        files_output_dir (str):         path to the directory where the patches and annotations will be stored.
        box_dims (dict):                dictionary specifying the dimensions of bounding boxes. Can be set manually in
                                        cases where the annotations contain arbitrary bbox dimensions and only the centers
                                        are reliable (as is allegedly the case in the Izembek dataset). Otherwise, 
                                        the box dimensions will be extracted from the annotations.
        radii (dict):                   dictionary containing the radii for each class. These values will be used as a
                                        distance-threshold (pixel-based Euclidean distance) beyond which detections 
                                        are considered false positive when running localization. 
        vis_output_dir (str):           path to the directory where the visualizations will be stored
    Returns: 
        a dictionary containing:    1.  a dictionary mapping patch names to metadata for all bounding boxes 
                                        that lie within the respective patches
                                    2.  the number of patches extracted from the image 
                                    3.  the number of annotations found in the image
                                    4.  the number of boxes that had to be clipped during the patching.
    """

    img_filename = Path(source_dir_img) / img["file_name"]
    pil_img = visutils.open_image(img_filename)
    assert pil_img.size[0] == img_width
    assert pil_img.size[1] == img_height

    if radii: 
        # make sure there is a radius for every class
        assert len(categories) == len(list(radii.keys())) and all(cat in radii.keys() for cat in categories), \
        f"Found classes {categories}, but raddi only for {list(radii.keys())}"

    annotations = img_id_to_ann[img["id"]]
    pos_patch_metadata_mapping_img = {}
    neg_patch_metadata_mapping_img = {}
    n_annotations_img = 0
    n_boxes_clipped_img = 0
    class_distr_img = {cat: 0 for cat in categories}
    empty_remaining = n_empties
 

    for patch in patch_start_positions:
        patch_coords = {"x_min": patch[PATCH_XSTART_IDX], 
                        "y_min": patch[PATCH_YSTART_IDX], 
                        "x_max": patch[PATCH_XSTART_IDX] + patch_dims["width"] - 1,
                        "y_max": patch[PATCH_YSTART_IDX] + patch_dims["height"] - 1}
        

        gt, patch_distr, n_boxes_clipped = get_annotations_in_patch(annotations_in_img=annotations, 
                                                                    boxes_in=boxes_in, 
                                                                    boxes_out=boxes_out, 
                                                                    patch_dims=patch_dims, 
                                                                    patch_coords=patch_coords, 
                                                                    categories=categories, 
                                                                    box_dims=box_dims)
        
        if not gt:
            if empty_remaining > 0:
                patch_name = patch_info2name(img["id"], patch_coords["x_min"], patch_coords["y_min"],
                                             is_empty=True)
                patch_ann_file = Path(files_output_dir) / f"{patch_name}.txt"
                
                patch_metadata = {
                    "patch_name": patch_name,
                    "original_image_id": img["id"],
                    "patch_x_min": patch_coords["x_min"],
                    "patch_y_min": patch_coords["y_min"],
                    "patch_x_max": patch_coords["x_max"],
                    "patch_y_max": patch_coords["y_max"]
                }
        
                neg_patch_metadata_mapping_img[patch_name] = patch_metadata

                empty_remaining -= 1
            else:
                continue
        else:
            n_boxes_clipped_img += n_boxes_clipped
            n_annotations_img += len(gt)
            for class_id in class_distr_img.keys():
                class_distr_img[class_id] += patch_distr[class_id]

            patch_name = patch_info2name(img["id"], patch_coords["x_min"], patch_coords["y_min"],
                                         is_empty=False)
            patch_ann_file = Path(files_output_dir) / f"{patch_name}.txt"
            

            patch_metadata = {
                "patch_name": patch_name,
                "original_image_id": img["id"],
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
        

        patch_image_file = Path(files_output_dir) / f"{patch_name}.jpg"
        assert not patch_image_file.exists()
        patch_im.save(patch_image_file, quality=patch_jpeg_quality)
        
        with open(patch_ann_file, 'w') as f:
            for ann in gt:
                if boxes_out:
                    ann_str = f"{ann[YOLO_BOX_CAT_IDX]} {ann[YOLO_BOX_XCENTER_IDX]} " \
                              f"{ann[YOLO_BOX_YCENTER_IDX]} {ann[YOLO_BOX_WIDTH_IDX]} " \
                              f"{ann[YOLO_BOX_HEIGHT_IDX]}\n"
                else:
                    ann_str = f"{ann[YOLO_PT_CAT_IDX]} {radii[ann[YOLO_PT_CAT_IDX]]} {ann[YOLO_PT_X_IDX]} {ann[YOLO_PT_Y_IDX]}\n"
                
                f.write(ann_str)
                
        assert Path.exists(patch_ann_file)
    
    if visualize:
        assert vis_output_dir, "Please provide a path to a directory where visulaizations can be stored!"
        vis_processed_img(img=img, source_dir_img=source_dir_img, img_id_to_ann=img_id_to_ann, 
                          patch_metadata_mapping_img=pos_patch_metadata_mapping_img,
                          patch_dims=patch_dims, boxes_out=boxes_out, output_dir=vis_output_dir)

    return {"pos_patches_mapping": pos_patch_metadata_mapping_img, 
            "neg_patches_mapping": neg_patch_metadata_mapping_img,
            "n_non_empty_patches":len(pos_patch_metadata_mapping_img.keys()),
            "n_empty_patches": len(neg_patch_metadata_mapping_img),
            "n_annotations": n_annotations_img, 
            "n_boxes_clipped":  n_boxes_clipped_img, 
            "class_distribution": class_distr_img}



def vis_processed_img(img: dict, source_dir_img: str, img_id_to_ann: dict, patch_metadata_mapping_img: dict, patch_dims: dict, boxes_out: bool, 
                      output_dir: str) -> None:
    """
    Draws annotations into the provided image and the corresponding patches. Image and patches are then stored in the specified 
    directory.
    Arguments:
        img (dict):                         dictionary containing the image metadata
        source_dir_img (str):               path to the directoryt where the image is stored
        img_id_to_ann (dict):               a dictionary mapping image ids to lists of annotations contained
                                            in the respective images. Bounding boxes are expected in the COCO-format.
        patch_metadata_mapping_img (dict):  dictionary containing the metadata for all patches of the image in question
        patch_dims (dict):                  dictionary containing the dimensions of patches
        boxes_out (bool):                   if false, the annotations are expected to contain point labels instead of bounding
                                            boxes
        output_dir (str):                   directory where the output files will be stroed

    Returns:
        None
    """
    
    output_path = Path(output_dir) / img["id"].split(".")[0]
    output_path.mkdir(parents=False, exist_ok=True)

    img_arr = cv2.imread(f"{source_dir_img}/{img['file_name']}")

    for ann in img_id_to_ann[img["id"]]:
        
        if boxes_out:
            #cv2 needs int coordinates
            xmin_img = round(ann["bbox"][COCO_BOX_XMIN_IDX])
            xmax_img = round(ann["bbox"][COCO_BOX_XMIN_IDX] + ann["bbox"][COCO_BOX_WIDTH_IDX])
            ymin_img = round(ann["bbox"][COCO_BOX_YMIN_IDX])
            ymax_img = round(ann["bbox"][COCO_BOX_YMIN_IDX] + ann["bbox"][COCO_BOX_HEIGHT_IDX])
         
            cv2.rectangle(img=img_arr, pt1=(xmin_img, ymin_img), pt2=(xmax_img, ymax_img), 
                          color=(0, 0, 255), thickness=1)
        else:
            x_center = round(ann["bbox"][COCO_BOX_XMIN_IDX] + ann["bbox"][COCO_BOX_WIDTH_IDX] / 2)
            y_center = round(ann["bbox"][COCO_BOX_YMIN_IDX] + ann["bbox"][COCO_BOX_HEIGHT_IDX] / 2)
            
            cv2.circle(img=img_arr, center=(x_center, y_center), radius=PT_VIS_CIRCLE_RADIUS, 
                       color=(0, 0, 255), thickness=-1)

    cv2.imwrite(str(output_path/f"full_img_{img['id']}.jpg"), img_arr)

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
                #convert relative (yolo-formatted) box data to absolute values for plotting
                x_center_absolute_patch = ann[YOLO_BOX_XCENTER_IDX] * patch_dims["width"]
                y_center_absolute_patch = ann[YOLO_BOX_YCENTER_IDX] * patch_dims["height"]
                width_absolute = ann[YOLO_BOX_WIDTH_IDX] * patch_dims["width"]
                height_absolute = ann[YOLO_BOX_HEIGHT_IDX] * patch_dims["height"]

                xmin_patch = round(x_center_absolute_patch - (width_absolute / 2))
                xmax_patch = round(x_center_absolute_patch + (width_absolute / 2))
                ymin_patch = round(y_center_absolute_patch - (height_absolute / 2))
                ymax_patch = round(y_center_absolute_patch + (height_absolute / 2))

                cv2.rectangle(img=patch_arr, pt1=(xmin_patch, ymin_patch), pt2=(xmax_patch, ymax_patch), 
                              color=(0, 255, 0), thickness=1)
            else:
                #convert relative coords to pixel for plotting
                x_patch = round(ann[YOLO_PT_X_IDX] * patch_dims["width"])
                y_patch = round(ann[YOLO_PT_Y_IDX] * patch_dims["height"])

                if x_patch == patch_dims["width"]:
                    x_patch -= 1
                if y_patch == patch_dims["height"]:
                    y_patch -= 1

                #check if circle fits into patch
                draw_circle = x_patch + PT_VIS_CIRCLE_RADIUS < patch_dims["width"] and \
                              x_patch - PT_VIS_CIRCLE_RADIUS > 0 and \
                              y_patch + PT_VIS_CIRCLE_RADIUS < patch_dims["height"] and \
                              y_patch - PT_VIS_CIRCLE_RADIUS > 0

                if draw_circle: 
                    cv2.circle(img=patch_arr, center=(x_patch, y_patch), radius=PT_VIS_CIRCLE_RADIUS, 
                               color=(0, 255, 0), thickness=-1)
                else:
                    patch_arr[y_patch, x_patch] = (0, 255, 0)

            cv2.imwrite(str(output_path / f"{patch_dict['patch_name']}.jpg"), patch_arr)


def generate_train_val_splits(pos_patches_mapping: dict, neg_patches_mapping: dict, val_frac: float, neg_frac: float, 
                              category_id_to_name: dict, base_dir: str, train_dir: str, val_dir: str, 
                              radii: dict = None) -> dict:
    """
    Generate the training-, and validation-split from a dictionary mapping patches to metadata. 
    The method takes a path to the base directory, where all the patch image-, and annotation-files are stored, 
    and moves them to the correct folders according to the generated splits. 
    Arguments: 
        pos_patches_mapping (dict):     dict mapping patch ids to metadata for patches that contain annotations 
        neg_patches_mapping (dict):     dict mapping patch ids to metadata for empty patches                    
        val_frac (float):               fraction of patches to be used for validation
        neg_frac (float):               fraction of negative (empty) samples to add to the training data
        category_id_to_name (dict):     dictionary mapping category ids to categpry names
        base_dir (str):                 path to the base directory
        train_dir (str):                path to the training data directory
        val_dir (str):                  path to the validation data directory
        radii (dict):                   dictionary containing the radii for each class. These values will be used as a
                                        distance-threshold (pixel-based Euclidean distance) beyond which detections 
                                        are considered false positive when running localization.
    Returns:
        Dictionary containing information (class distribution and size) about the splits
    """

    pos_patches_ids = list(pos_patches_mapping.keys())
    neg_patches_ids = list(neg_patches_mapping.keys())

    n_val_patches = int(val_frac*len(pos_patches_ids))
    n_train_patches = len(pos_patches_ids) - n_val_patches

    # add correct amount of negative samples if possible (enough empty patches)
    desired_total = int(math.floor(n_train_patches / (1 - neg_frac)))
    n_negs = desired_total - n_train_patches
    random.shuffle(neg_patches_ids)
    train_patches_ids_neg = neg_patches_ids[:n_negs] if n_negs < len(neg_patches_ids) else neg_patches_ids
    negs_not_used = [neg_id for neg_id in neg_patches_ids if neg_id not in train_patches_ids_neg]

    # create splits
    random.shuffle(pos_patches_ids)
    val_patches_ids = pos_patches_ids[:n_val_patches]
    train_patches_ids = pos_patches_ids[n_val_patches:]
    train_patches_ids.extend(train_patches_ids_neg)

    with open(f"{train_dir}/train_mapping.json","w") as f:
        json.dump(train_patches_ids,f,indent=1)
        
    with open(f"{val_dir}/val_mapping.json","w") as f:
        json.dump(val_patches_ids,f,indent=1)


    # Copy annotation files to train/val/test folders and collect split statistics
    train_distribution = {cat: 0 for cat in category_id_to_name.keys()}
    val_distribution = {cat: 0 for cat in category_id_to_name.keys()}


    # For each patch
    for patch_name in tqdm((train_patches_ids + val_patches_ids), total=len(train_patches_ids + val_patches_ids)):
        
        # Make sure we have an annotation file
        src_path_ann = f"{base_dir}/{patch_name}.txt"
        src_path_img = f"{base_dir}/{patch_name}.jpg"
        
        assert Path.exists(Path(src_path_ann)) and Path.exists(Path(src_path_img))
        
        # Copy files to the place it belongs and collect class distributions
        if patch_name in train_patches_ids:
            if "empty" not in patch_name:
                for class_id in train_distribution.keys():
                    train_distribution[class_id] += pos_patches_mapping[patch_name]["class_distribution"][class_id]
            target_folder = train_dir
        elif patch_name in val_patches_ids:
            for class_id in val_distribution.keys():
                val_distribution[class_id] += pos_patches_mapping[patch_name]["class_distribution"][class_id]
            target_folder = val_dir

        target_path_ann = f"{target_folder}/{Path(src_path_ann).name}"
        target_path_img = f"{target_folder}/{Path(src_path_img).name}"
        shutil.move(src_path_ann, target_path_ann)
        shutil.move(src_path_img, target_path_img)

    #Generate the YOLO training dataset file
    with open(f"{base_dir}/dataset.yaml","w") as f:
        train_dir_rel = Path(train_dir).relative_to(base_dir)
        val_dir_rel = Path(val_dir).relative_to(base_dir)
        
        f.write("# Train/val/test sets\n" \
                f"path: {base_dir}\n" \
                f"train: {train_dir_rel}\n" \
                f"val: {val_dir_rel}\n" \
                "\n" \
                "# Classes\n" \
                "names:\n")
        
        for class_id,class_name in category_id_to_name.items():
            f.write(f"  {class_id}: {class_name.strip()}\n")
        if radii: 
            f.write("\nradii:\n")
            for class_id, radius in radii.items():
                f.write(f"  {class_id}: {radius}\n")

    return {"train": {"distribution": train_distribution, "size": len(train_patches_ids), "empty": len(train_patches_ids_neg)},
            "val": {"distribution": val_distribution, "size": len(val_patches_ids)}}, negs_not_used

