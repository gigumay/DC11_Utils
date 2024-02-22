import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")

import math
import random
import cv2

from pathlib import Path

import visualization_utils as visutils


YOLO_BOX_CAT_IDX = 0
YOLO_BOX_XCENTER_IDX = 1
YOLO_BOX_YCENTER_IDX = 2
YOLO_BOX_WIDTH_IDX = 3
YOLO_BOX_HEIGHT_IDX = 4

YOLO_PT_X_IDX = 0
YOLO_PT_Y_IDX = 1
YOLO_PT_CAT_IDX = 2

COCO_BOX_XMIN_IDX = 0
COCO_BOX_YMIN_IDX = 1
COCO_BOX_WIDTH_IDX = 2
COCO_BOX_HEIGHT_IDX = 3

INPUT_FORMAT_Y_IDX = 0
INPUT_FORMAT_X_IDX = 1

PATCH_XSTART_IDX = 0
PATCH_YSTART_IDX = 1


def randomize_centers(yolo_box: list, buffer_dims_pct: float) -> dict:
    """
    Given a yolo-formatted bounding box, a point that lies within a buffer around the box center point
    is randomly sampled and returned.
    Arguments:
        yolo_box (list):            list specifying the yolo-formatted bounding box
        buffer_dims_pct (float):    percentage of the box-width and -height that is going to be used to 
                                    create the buffer. Specifically, the buffer will have its center at 
                                    the box center and will have the dimensions (box_width * buffer_dims_pct), 
                                    (box_height * buffer_dims_pct).
    Returns:
        A dictionary containing the randomly sampled point coordinates and the ground truth class
    """
    buffer_width = yolo_box[YOLO_BOX_WIDTH_IDX] * buffer_dims_pct
    buffer_height = yolo_box[YOLO_BOX_HEIGHT_IDX] * buffer_dims_pct

    random_x = random.randint(math.floor(yolo_box[YOLO_BOX_XCENTER_IDX] - buffer_width), 
                              math.ceil(yolo_box[YOLO_BOX_XCENTER_IDX] + buffer_width))
    random_y = random.randint(math.floor(yolo_box[YOLO_BOX_YCENTER_IDX] - buffer_height),
                              math.ceil(yolo_box[YOLO_BOX_YCENTER_IDX] + buffer_height))

    return {"x": random_x, "y": random_y, "category_id": yolo_box[YOLO_BOX_CAT_IDX]}



def get_patch_start_positions(img_width: int, img_height: int, patch_dims: dict, overlap: float) -> list: 
    """
    Creates list of the starting positions (in pixel offsets) of each patch in a given image. 
    Arguments:
        img_width (int):    width of the image in pixels
        img_height (int):   height of the image in pixels
        patch_dims (dict):  Dict containing the dimensions of the patches.
    Returns:
        list of starting positions
    """
    patch_start_positions = []
    
    # account for overlap between patches
    overlap_pxs_x = math.ceil(patch_dims["width"] * overlap)
    overlap_pxs_y = math.ceil(patch_dims["height"] * overlap)


    n_x_patches = img_width // (patch_dims["width"] - overlap_pxs_x)
    # account for additional patch at the edges
    if img_width - ((patch_dims["width"] - overlap_pxs_x) * n_x_patches) != 0:
        n_x_patches += 1

    n_y_patches = img_height // (patch_dims["height"] - overlap_pxs_y)
    if img_height - ((patch_dims["height"] - overlap_pxs_y) * n_y_patches) != 0:
        n_y_patches += 1

    for i_x_patch in range(n_x_patches):
        x_start = (patch_dims["width"] - overlap_pxs_x) * i_x_patch
        x_end = x_start + patch_dims["width"] - 1
        # if patch overshoots: shift starting point
        if x_end >= img_width:
            assert i_x_patch == n_x_patches - 1
            overshoot = (x_end - img_width) + 1
            x_start -= overshoot

        for i_y_patch in range(n_y_patches):
            y_start = (patch_dims["height"] - overlap_pxs_y) * i_y_patch
            y_end = y_start + patch_dims["height"] - 1
            if y_end >= img_height:
                assert i_y_patch == n_y_patches - 1
                overshoot =  (y_end - img_height) + 1
                y_start -= overshoot
            patch_start_positions.append([x_start,y_start])

    assert patch_start_positions[-1][PATCH_XSTART_IDX] + patch_dims["width"] == img_width
    assert patch_start_positions[-1][PATCH_YSTART_IDX] + patch_dims["height"] == img_height

    return patch_start_positions
    

def patch_info_to_patch_name(image_name: str, patch_x_min: int, patch_y_min: int) -> str:
    """
    Generates the name of a patch.
    Arguments: 
        image_name (str):   name of the image that contains the patch
        patch_x_min (int):  x coordinate of the left upper corner of the patch within the 
                            original image
        patch_y_min (int):  y coordinate of the left upper corner of the patch within the 
                            original image 
    Returns: 
        string containing the patch name
    """
    patch_name = f"{image_name}_{str(patch_x_min).zfill(4)}_{str(patch_y_min).zfill(4)}"
    return patch_name


def get_boxes_in_patch_img_lvl(annotations_in_img: list, patch_coords: dict, box_dims: dict = None) -> list:
    """
    For a given patch, create a list of dictionaries that for each annotation in the patch contain the
    bounding box center coordinates (at image level), the bounding box dimensions and the class of the annotation.
    Annotations are expected to be in the COCO-format.
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
        
        patch_contains_box = (patch_coords["x_min"] < box_x_max and box_x_min < patch_coords["x_max"] \
                             and patch_coords["y_min"] < box_y_max and box_y_min < patch_coords["y_max"])
        
        if patch_contains_box:
            patch_boxes.append({"x_center": box_x_center, "y_center": box_y_center, "box_dims": box_dims_checked,
                                "category_id": ann['category_id']})
    
    return patch_boxes



def get_boxes_at_patch_lvl(annotations_in_img: list, clip_boxes: bool, patch_dims: dict, patch_coords: dict, 
                           categories: list, box_dims: dict = None) -> tuple[list, dict, list, int]:
    """
    For a given patch, create a list of athe bounding boxes (as yolo-formatted dictionaries and with 
    patch-level coordinates) that lie within that patch.
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
        clip_boxes (bool):          whether or not to clip boxes that exceed the patch boundaries
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories (list):          list of classes in the datset
        box_dims (dict):            dictionary specifying the dimensions of bounding boxes. If None, 
                                    the box dimensions will be extracted from the annotations.
    Returns: 
        1.  a list containing YOLO-formatted bounding boxes that lie within the provided patch 
            (coordinates at patch-level). 
        2.  the class distribution in the patch
        3.  a list containing  dictionaries with the coordinates (in pixe and at patch-level) of 
            the bbox centers
            that lie within the provided patch.
        4.  the number of boxes that were clipped 
    """

    class_distr_patch = {cat["id"]: 0 for cat in categories}
    patch_boxes = get_boxes_in_patch_img_lvl(annotations_in_img=annotations_in_img, 
                                             patch_coords=patch_coords, box_dims=box_dims)
    
    if not patch_boxes:
        return [], [], 0, class_distr_patch
    
    n_clipped_boxes = 0
    yolo_boxes_patch = []
    patch_box_centers = []

    for box_dict in patch_boxes:
        x_center_absolute_original = box_dict["x_center"]
        y_center_absolute_original = box_dict["y_center"]
        
        x_center_absolute_patch = x_center_absolute_original - patch_coords["x_min"]
        y_center_absolute_patch = y_center_absolute_original - patch_coords["y_min"]
        
        assert (1 + patch_coords["x_max"] - patch_coords["x_min"]) == patch_dims["width"]
        assert (1 + patch_coords["y_max"] - patch_coords["y_min"]) == patch_dims["height"]
        
        x_center_relative = x_center_absolute_patch / patch_dims["width"]
        y_center_relative = y_center_absolute_patch / patch_dims["height"]
        
        # if desired: clip boxes that are not entirely contained in a given patch 
        if clip_boxes:
            clipped_box = False

            # box size relative to patch size
            box_dims_rel = {"width": box_dict["box_dims"]["width"] / patch_dims["width"], 
                            "height": box_dict["box_dims"]["height"] / patch_dims["height"]}
            
            box_right = x_center_relative + (box_dims_rel["width"] / 2.0)                    
            if box_right > 1.0:
                clipped_box = True
                overhang = box_right - 1.0
                box_dims_rel["width"] -= overhang
                x_center_relative -= overhang / 2.0

            box_bottom = y_center_relative + (box_dims_rel["height"] / 2.0)                                        
            if box_bottom > 1.0:
                clipped_box = True
                overhang = box_bottom - 1.0
                box_dims_rel["height"] -= overhang
                y_center_relative -= overhang / 2.0
            
            box_left = x_center_relative - (box_dims_rel["width"] / 2.0)
            if box_left < 0.0:
                clipped_box = True
                overhang = abs(box_left)
                box_dims_rel["width"] -= overhang
                x_center_relative += overhang / 2.0
                
            box_top = y_center_relative - (box_dims_rel["height"] / 2.0)
            if box_top < 0.0:
                clipped_box = True
                overhang = abs(box_top)
                box_dims_rel["height"] -= overhang
                y_center_relative += overhang / 2.0
                
            if clipped_box:
                n_clipped_boxes += 1
        
        # YOLO annotations are category, x_center, y_center, w, h
        yolo_box = [box_dict["category_id"], x_center_relative, y_center_relative, 
                    box_dims_rel["width"], box_dims_rel["height"]]
        
        yolo_boxes_patch.append(yolo_box)
        patch_box_centers.append[{"x": box_dict["x_center"], "y": box_dict["y_center"]}]
        class_distr_patch[box_dict["category_id"]] += 1

    return yolo_boxes_patch, class_distr_patch, patch_box_centers, n_clipped_boxes



def get_points_in_patch(annotations_in_img: list, patch_dims: dict, patch_coords: dict, categories: list) -> tuple[list, dict]:
    """
    For a given patch, create a list of point labels (as yolo-formatted dictionaries) that lie within that patch
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (point labels) in the image the patch
                                    was taken from.
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories (list):          list of classes in the datset
    Returns: 
        1.  a list containing  dictionaries with the coordinates (in pixel and at patch level) of the point labels that
            lie within the provided patch, as well as the corresponding class.
        2.  the class distribution in the patch
    """

    class_distr_patch = {cat["id"]: 0 for cat in categories}
    gt_points = []
            
    assert (1 + patch_coords["x_max"] - patch_coords["x_min"]) == patch_dims["width"]
    assert (1 + patch_coords["y_max"] - patch_coords["y_min"]) == patch_dims["height"]

    for ann in annotations_in_img:                
        gt_x = ann["x"]
        gt_y = ann["y"]
        
        patch_contains_pt = (patch_coords["x_min"] < gt_x and gt_x < patch_coords["x_max"] \
                             and patch_coords["y_min"] < gt_y and gt_y < patch_coords["y_max"])
        
        if patch_contains_pt:
            x_center_absolute_patch = gt_x - patch_coords["x_min"]
            y_center_absolute_patch = gt_y - patch_coords["y_min"]
    
            #x_center_relative = x_center_absolute_patch / patch_dims["width"]
            #y_center_relative = y_center_absolute_patch / patch_dims["height"]
            
            gt_points.append([x_center_absolute_patch, y_center_absolute_patch, ann["category_id"]])
            class_distr_patch[ann['category_id']] += 1
    
    return gt_points, class_distr_patch



def get_annotations_in_patch(annotations_in_img: list, boxes_in: bool, boxes_out: bool, patch_dims: dict, 
                             patch_coords: dict, categories: list, box_dims: dict, buffer_dims_pct: float, 
                             clip_boxes: bool)-> tuple[list, dict, tuple[list, None], tuple[list, None]]:
    """
    Retrieves the annotations in a given patch. If the ground truth contains bounding boxes, this method can return 
    point labels that are randomly sampled within a given buffer around the box centers.
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
        buffer_dims_pct (float):    percentage of the box-width and -height that is going to be used to 
                                    create the buffer to sample point labels from. Specifically, the buffer will have
                                    its center at the box center and will have the dimensions 
                                    (box_width * buffer_dims_pct), (box_height * buffer_dims_pct).
        clip_boxes (bool):          whether or not to clip boxes that exceed the patch boundaries
        
    """

    # sanity check
    assert not (not boxes_in and boxes_out)

    if boxes_in: 
        box_centers, yolo_boxes, n_clipped_boxes, class_distr = get_boxes_at_patch_lvl(annotations_in_img=annotations_in_img,
                                                                                       clip_boxes=clip_boxes, 
                                                                                       patch_dims=patch_dims,
                                                                                       patch_coords=patch_coords, 
                                                                                       categories=categories,
                                                                                       box_dims=box_dims)
        if boxes_out:
            return yolo_boxes, class_distr, box_centers,  n_clipped_boxes
        else:
            gt_points = [randomize_centers(yolo_box=box, buffer_dims_pct=buffer_dims_pct) for box in yolo_boxes]
            return gt_points, class_distr, None, None
    else: 
         gt_points, class_distr = get_points_in_patch(annotations_in_img=annotations_in_img, patch_dims=patch_dims, 
                                                      patch_coords=patch_coords, categories=categories)
         return gt_points, class_distr, None, None
        
    


def process_image(source_dir_img: str, img: dict, img_width: int, img_height: int, is_negative: bool, boxes_in: bool,
                  boxes_out: bool, img_id_to_ann: dict, patch_dims: dict, patch_start_positions: list, 
                  patch_jpeg_quality: int, write_empty_file_neg: bool, categories: list, visualize: bool, 
                  dest_dir_imgs: tuple[None, str], dest_dir_txt: str, box_dims: dict = None, 
                  buffer_dims_pct: float = 0.25,  clip_boxes: bool = True, vis_output_dir: str = None) -> dict:
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
        is_negative (bool):             set to true if the image does not contain animals (is empty)
        boxes_in (bbol):                must be set to true if the input annotations contain bounding boxes
        boxes_out (bool):               if false, return point labels instead of bounding boxes. Must be set to false
                                        when working iwht point labels. 
        img_id_to_ann (dict):           a dictionary mapping image ids to lists of annotations contained
                                        in the respective images. bounding boxes are expected in the COCO-format.
        patch_dims (dict):              dict specifying the dimensions of the patches.
        patch_start_positions (list):   list of pixel coordinates specifying the starting positions of the 
                                        patches
        patch_jpeg_quality (int):       quality of the patch-images
        write_empty_file_neg (bool):    if true, empty yolo-files are created for empty patches
        categories:                     list of classes in the datset
        visualize (bool):               if true, the annotations are drawn into the image and the patches,
                                        which are then written into a specified directory.
        dest_dir_imgs (str):            path to the directory the patch images will be stored in. If None, patches won't
                                        be stored as images
        dest_dir_txt (str):             path to the directory the bounding-box metadat files will be stored in
        box_dims (dict):                dictionary specifying the dimensions of bounding boxes. Can be set manually in
                                        cases where the annotations contain arbitrary bbox dimensions and only the centers
                                        are reliable (as is allegedly the case in the Izembek dataset). Otherwise, 
                                        the box dimensions will be extracted from the annotations.
        buffer_dims_pct (float):        percentage of the box-width and -height that is going to be used to 
                                        create the buffer to sample point labels from when point annotations 
                                        are to be extracted from bounding boxes
        clip_boxes (bool):              whether or not to clip boxes that exceed patch boundaries
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

    annotations = img_id_to_ann[img["id"]]
    patch_metadata_mapping_img = {}
    n_annotations_img = 0
    n_boxes_clipped_img = 0
    n_patches_img = 0
    class_distr_img = {cat["id"]: 0 for cat in categories}

    for patch in patch_start_positions:
        patch_coords = {"x_min": patch[PATCH_XSTART_IDX], 
                        "y_min": patch[PATCH_YSTART_IDX], 
                        "x_max": patch[PATCH_XSTART_IDX] + patch_dims["width"] - 1,
                        "y_max": patch[PATCH_YSTART_IDX] + patch_dims["height"] - 1}
        

        if not is_negative:
            gt, patch_distr, box_centers, n_boxes_clipped = get_annotations_in_patch(annotations_in_img=annotations, 
                                                                                     boxes_in=boxes_in, 
                                                                                     boxes_out=boxes_out, 
                                                                                     patch_dims=patch_dims, 
                                                                                     patch_coords=patch_coords, 
                                                                                     categories=categories, 
                                                                                     box_dims=box_dims, 
                                                                                     buffer_dims_pct=buffer_dims_pct,
                                                                                     clip_boxes=clip_boxes)
            
            # skip empty patches in positive images
            if not gt:
                continue 

            n_boxes_clipped_img += n_boxes_clipped
            n_annotations_img += len(gt)
            class_distr_img += patch_distr

        
        patch_name = patch_info_to_patch_name(img["file_name"], patch_coords["x_min"], patch_coords["y_min"])
        patch_image_file = Path(dest_dir_imgs) / f"{patch_name}.jpg"
        patch_ann_file = Path(dest_dir_txt) / f"{patch_name}.txt"
        
        assert not patch_image_file.exists()
        assert not patch_ann_file.exists()
        
        patch_metadata = {
           "patch_name": patch_name,
           "original_image_id": img["id"],
           "patch_x_min": patch_coords["x_min"],
           "patch_y_min": patch_coords["y_min"],
           "patch_x_max": patch_coords["x_max"],
           "patch_y_max": patch_coords["y_max"],
           "boxes": gt if boxes_out else None,
           "points": gt if not boxes_out else None,
           "class_distribution": patch_distr,
           "hard_negative": is_negative
        }

        if not is_negative and boxes_out:
            patch_metadata["box_centers"] = box_centers

        patch_metadata_mapping_img[patch_name] = patch_metadata

        if dest_dir_imgs:
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
  
            patch_im.save(patch_image_file, quality=patch_jpeg_quality)
        
        if not is_negative or write_empty_file_neg:
            with open(patch_ann_file, 'w') as f:
                for ann in gt:
                    if boxes_out:
                        ann_str = f"{ann[YOLO_BOX_CAT_IDX]} {ann[YOLO_BOX_XCENTER_IDX]} " \
                                  f"{ann[YOLO_BOX_YCENTER_IDX]} {ann[YOLO_BOX_WIDTH_IDX]} " \
                                  f"{ann[YOLO_BOX_HEIGHT_IDX]}\n"
                    else:
                        ann_str = f"{ann[YOLO_BOX_CAT_IDX]} {ann[YOLO_PT_X_IDX]} {ann[YOLO_PT_Y_IDX]}\n"
                    
                    f.write(ann_str)
                    
            assert Path.exists(patch_ann_file)

        n_patches_img += 1
    
    if visualize:
        assert vis_output_dir, "Please provide a path to a directory where visulaizations can be stored!"
        vis_processed_img(img=img, source_dir_img=source_dir_img, img_id_to_ann=img_id_to_ann, patch_metadata_mapping_img=patch_metadata_mapping_img,
                          output_dir=vis_output_dir)

    return {"patches_mapping": patch_metadata_mapping_img, "n_patches": n_patches_img,
            "n_annotations": n_annotations_img, "n_boxes_clipped":  n_boxes_clipped, 
            "class_distribution": class_distr_img}



def vis_processed_img(img: dict, source_dir_img: str, img_id_to_ann: dict, patch_metadata_mapping_img: dict, boxes_out: bool, 
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
        boxes_out (bool):                   if false, the annotations are expected to contian point labels instead of boundin
                                            boxes
        output_dir (str):                   directory where the output files will be stroed

    Returns:
        None
    """
    
    output_path = Path(output_dir) / img["file_name"]
    output_path.mkdir(parents=False, exist_ok=True)

    img_arr = cv2.imread(Path(source_dir_img) / img["file_name"])

    for ann in img_id_to_ann[img["id"]]:
        
        if boxes_out:
            xmin_img = ann["bbox"][COCO_BOX_XMIN_IDX]
            xmax_img = ann["bbox"][COCO_BOX_XMIN_IDX] + ann["bbox"][COCO_BOX_WIDTH_IDX]
            ymin_img = ann["bbox"][COCO_BOX_YMIN_IDX]
            ymax_img = ann["bbox"][COCO_BOX_YMIN_IDX] + ann["bbox"][COCO_BOX_HEIGHT_IDX]
         
            cv2.rectangle(img_arr, (xmin_img, ymin_img), (xmax_img, ymax_img), (0, 255, 0), 1)
        else:
            img_arr[ann["point"][INPUT_FORMAT_Y_IDX], ann["point"][INPUT_FORMAT_X_IDX]] = (0, 255, 0)

    cv2.imwrite(str(output_path / f"full_img.jpg"))

    for key in patch_metadata_mapping_img:
        patch_dict = patch_metadata_mapping_img[key]
        patch_arr = img_arr[patch_dict["patch_y_min"] : patch_dict["patch_y_max"], patch_dict["patch_x_min"], 
                            patch_dict["patch_x_max"]]

        if boxes_out:
            gt = patch_dict["boxes"]
        else:
            gt = patch_dict["points"]

        for ann in gt:
            if boxes_out:
                xmin_patch = ann[YOLO_BOX_XCENTER_IDX] - ann[YOLO_BOX_WIDTH_IDX] / 2
                xmax_patch = ann[YOLO_BOX_XCENTER_IDX] + ann[YOLO_BOX_WIDTH_IDX] / 2
                ymin_patch = ann[YOLO_BOX_YCENTER_IDX] - ann[YOLO_BOX_HEIGHT_IDX] / 2
                ymax_patch = ann[YOLO_BOX_YCENTER_IDX] + ann[YOLO_BOX_HEIGHT_IDX] / 2

                cv2.rectangle(patch_arr, (xmin_patch, ymin_patch), (xmax_patch, ymax_patch), (0, 255, 0), 1)
            else:
                patch_arr[ann["y"], ann["x"]] = (0, 255, 0)

            cv2.imwrite(str(output_path / f"{patch_dict['patch_name']}.jpg"))

#TODO: Add funtionality to add empty patches so that negative sampling doesn't rely on images beigng annotated as empty