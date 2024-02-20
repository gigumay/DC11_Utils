import sys
sys.path.append("/home/giacomo/programs/MegaDetector/md_visualization")

import math
import cv2

from pathlib import Path

import visualization_utils as visutils


YOLO_BOX_CAT_IDX = 0
YOLO_BOX_XCENTER_IDX = 1
YOLO_BOX_YCENTER_IDX = 2
YOLO_BOX_WIDTH_IDX = 3
YOLO_BOX_HEIGHT_IDX = 4

COCO_BOX_XMIN_IDX = 0
COCO_BOX_YMIN_IDX = 1
COCO_BOX_WIDTH_IDX = 2
COCO_BOX_HEIGHT_IDX = 3

PATCH_XSTART_IDX = 0
PATCH_YSTART_IDX = 1

BOX_CENTERS_DICT_IDX = 0
BOX_CAT_IDX = 1



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


def get_box_centers_in_patch(annotations_in_img: list, box_dims: dict, patch_coords: dict) -> list:
    """
    For a given patch, create a list of the coordinates of all the centers of annotation boxes that 
    lie within that patch. Annotations are expected to be in the COCO-format.
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
        box_dims (dict):            dicitonary specifying the dimensions of bounding boxes.
        patch_coords (dict):        dictionary specifying the coordinates (in pixel) of the 
                                    patch in question
    Returns: 
        A list containing dictironaries with the coordinates (in pixel) of the centers of the boxes that lie within 
        the provided patch, as well as the corresponding category. 
    """
    patch_box_centers = []
    for ann in annotations_in_img:                
        # In the input annotations, boxes are expected to x/y/w/h
        box_x_center = ann['bbox'][COCO_BOX_XMIN_IDX] + (ann['bbox'][COCO_BOX_WIDTH_IDX]/2.0)
        box_y_center = ann['bbox'][COCO_BOX_YMIN_IDX] + (ann['bbox'][COCO_BOX_HEIGHT_IDX]/2.0)
        
        box_x_min = box_x_center - (box_dims["width"]/2.0)
        box_x_max = box_x_center + (box_dims["width"]/2.0)
        box_y_min = box_y_center - (box_dims["height"]/2.0)
        box_y_max = box_y_center + (box_dims["height"]/2.0)
        
        patch_contains_box = (patch_coords["x_min"] < box_x_max and box_x_min < patch_coords["x_max"] \
                             and patch_coords["y_min"] < box_y_max and box_y_min < patch_coords["y_max"])
        
        if patch_contains_box:
            patch_box_centers.append([{"x": box_x_center, "y": box_y_center}, ann['category_id']])
    
    return patch_box_centers


def get_boxes_in_patch(annotations_in_img: list, clip_boxes: bool, box_dims: dict, patch_dims: list, 
                       patch_coords: dict, categories: list) -> tuple[list, list, int, dict]:
    """
    For a given patch, create a list of annotation boxes (as yolo-formatted dictionaries) that 
    lie within that patch
    Arguments: 
        annotations_in_img (list):  list containing dictionaries of annotations (bounding boxes)
                                    in the image the patch was taken from.
        clip_boxes (bool):          whether or not to clip boxes that exceed the patch boundaries
        box_dims (dict):            dict specifying the dimensions of bounding boxes.
        patch_dims (dict):          dict specifying the dimensions of the patch.
        patch_coords (dict):        dict specifying the coordinates (in pixel) of the patch
        categories:                 list of classes in the datset
    Returns: 
        1.  a list containing  dictionaries with the coordinates (in pixel) of the bbox centers of the boxes that lie within 
            the provided patch.
        2.  a list containing YOLO-formatted bounding boxes that lie within the provided patch. 
        3.  the number of boxes that were clipped 
        4.  the class distribution in the patch
    """

    class_distr_patch = {cat["id"]: 0 for cat in categories}
    patch_box_centers = get_box_centers_in_patch(annotations_in_img=annotations_in_img, 
                                                 box_dims=box_dims, patch_coords=patch_coords)
    
    if not patch_box_centers:
        return [], [], 0
    
    n_clipped_boxes = 0
    yolo_boxes_patch = []

    for coords_cat in patch_box_centers:
        x_center_absolute_original = coords_cat[BOX_CENTERS_DICT_IDX]["x"]
        y_center_absolute_original = coords_cat[BOX_CENTERS_DICT_IDX]["y"]
        
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
            box_dims_rel = {"width": box_dims["width"] / patch_dims["width"], "height": box_dims["height"] / patch_dims["height"]}
            
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
        yolo_box = [coords_cat[BOX_CAT_IDX], x_center_relative, y_center_relative, 
                    box_dims_rel["width"], box_dims_rel["height"]]
        
        yolo_boxes_patch.append(yolo_box)
        class_distr_patch[coords_cat[BOX_CAT_IDX]] += 1

    return patch_box_centers, yolo_boxes_patch, n_clipped_boxes, class_distr_patch


def process_image(source_dir_img: str, img: dict, img_width: int, img_height: int, is_negative: bool, 
                  img_id_to_ann: dict, patch_dims: dict, patch_start_positions: list, patch_jpeg_quality: int, 
                  box_dims: dict, clip_boxes: bool,  write_tiles: bool, write_empty_file_neg: bool, categories: list,
                  visualize: bool, dest_dir_imgs: str, dest_dir_txt: str, vis_output_dir: str = None) -> dict:
    """
    Process a given image. Processing consists of dividing the image into patches and assigning each 
    patch a set of bounding boxes that lie within it. If the corresponding parameters are set accordingly,
    the patches are saved as image files and bounding box-metadata is written to files in yolo-format (happens
    always for patches that contain bounding boxes, but if desired empty files can be written for empty patches
    as well). 
    Arguments: 
        source_dir_img (str):           path to the directoryt where the image is stored
        img (dict):                     dictionary containing the image metadata
        img_width (int):                width of the image
        img_height (int):               height of the image
        is_negative (bool):             set to true if the image does not contain animals (is empty)
        img_id_to_ann (dict):           a dictionary mapping image ids to lists of bounding boxes contained
                                        in the respective images. Annotations are expected in the COCO-format.
        patch_dims (dict):              dict specifying the dimensions of the patches.
        patch_start_positions (list):   list of pixel coordinates specifying the starting positions of the 
                                        patches
        patch_jpeg_quality (int):       quality of the patch-images
        box_dims (dict):                dict specifying the dimensions of bounding boxes.
        clip_boxes (bool):              whether or not to clip boxes that exceed patch boundaries
        write_tiles (bool):             if true, the patches are saved as images 
        write_empty_file_neg (bool):    if true, empty yolo-files are created for empty patches
        categories:                     list of classes in the datset
        visualize (bool):               if true, the bounding boxes are drawn into the image and the patches,
                                        which are then written into a specified directory.
        dest_dir_imgs (str):            path to the directory the patch images will be stored in
        dest_dir_txt (str):             path to the directory the bounding-box metadat files will be stored in
        vis_output_dir (str):           path to the directory where the visualizations will be stored
    Returns: 
        a dictionary containing:    1.  a dictionary mapping patch names to metadata for all bounding boxes 
                                        that lie within the respective patches
                                    2.  the number of patches extracted from the image 
                                    3.  the number of boxes found in the image
                                    4.  the number of boxes that had to be clipped during the patching.
    """

    img_filename = Path(source_dir_img) / img["file_name"]
    pil_img = visutils.open_image(img_filename)
    assert pil_img.size[0] == img_width
    assert pil_img.size[1] == img_height

    annotations = img_id_to_ann[img["id"]]
    patch_metadata_mapping_img = {}
    n_boxes_img = 0
    n_boxes_clipped_img = 0
    n_patches = 0
    
    class_distr_img = {cat["id"]: 0 for cat in categories}

    for patch in patch_start_positions:
        patch_coords = {"x_min": patch[PATCH_XSTART_IDX], 
                        "y_min": patch[PATCH_YSTART_IDX], 
                        "x_max": patch[PATCH_XSTART_IDX] + patch_dims["width"] - 1,
                        "y_max": patch[PATCH_YSTART_IDX] + patch_dims["height"] - 1}
        
        box_centers = []
        boxes = []
        n_boxes_clipped = 0

        if not is_negative:
            box_centers, boxes, n_boxes_clipped, patch_distr = get_boxes_in_patch(annotations_in_img=annotations, 
                                                                                  clip_boxes=clip_boxes, box_dims=box_dims, 
                                                                                  patch_dims=patch_dims, patch_coords=patch_coords, 
                                                                                  categories=categories)
            
            # skip empty patches in positive images
            if not box_centers:
                continue 

            n_boxes_clipped_img += n_boxes_clipped
            n_boxes_img += len(boxes)
            class_distr_img += patch_distr

        
        patch_name = patch_info_to_patch_name(img["file_name"], patch_coords["x_min"], patch_coords["y_min"])
        patch_image_file = Path(dest_dir_imgs) / f"{patch_name}.jpg"
        patch_ann_file = Path(dest_dir_txt) / f"{patch_name}.txt"
        
        assert not patch_image_file.exists()
        assert not patch_ann_file.exists()
        
        patch_metadata = {
           'patch_name': patch_name,
           'original_image_id': img["id"],
           'patch_x_min': patch_coords["x_min"],
           'patch_y_min': patch_coords["y_min"],
           'patch_x_max': patch_coords["x_max"],
           'patch_y_max': patch_coords["y_max"],
           'boxes': boxes,
           'class_distribution': patch_distr,
           'hard_negative': is_negative
        }

        if not is_negative:
            patch_metadata["box_centers"] = box_centers

        patch_metadata_mapping_img[patch_name] = patch_metadata

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
        
        # Write out patch image
        if write_tiles:
            patch_im.save(patch_image_file, quality=patch_jpeg_quality)
        
        if not is_negative or write_empty_file_neg:
            with open(patch_ann_file, 'w') as f:
                for yolo_box in boxes:
                    f.write(f"{yolo_box[YOLO_BOX_CAT_IDX]} {yolo_box[YOLO_BOX_XCENTER_IDX]} {yolo_box[YOLO_BOX_YCENTER_IDX]} " \
                            f"{yolo_box[YOLO_BOX_WIDTH_IDX]} {yolo_box[YOLO_BOX_HEIGHT_IDX]}\n")
                    
            assert Path.exists(patch_ann_file)

        n_patches += 1
    
    if visualize:
        vis_processed_img(img=img, source_dir_img=source_dir_img, img_id_to_ann=img_id_to_ann, patch_metadata_mapping_img=patch_metadata_mapping_img,
                          output_dir=vis_output_dir)

    return {"patches_mapping": patch_metadata_mapping_img, "n_patches": n_patches,
            "n_boxes": n_boxes_img, "n_boxes_clipped":  n_boxes_clipped}




def vis_processed_img(img: dict, source_dir_img: str, img_id_to_ann: dict, patch_metadata_mapping_img: dict, output_dir: str) -> None:
    """
    Draws bounding boxes into the provided image and the corresponding patches. Image and patches are then written into the specified 
    directory.
    Arguments:
        img (dict):                         dictionary containing the image metadata
        source_dir_img (str):               path to the directoryt where the image is stored
        img_id_to_ann (dict):               a dictionary mapping image ids to lists of bounding boxes contained
                                            in the respective images. Annotations are expected in the COCO-format.
        patch_metadata_mapping_img (dict):  dictionary containing the metadat for all patches of the image in question
        output_dir (str):                   directory where the output files will be stroed

    Returns:
        None
    """
    
    output_path = Path(output_dir) / img["file_name"]
    output_path.mkdir(parents=False, exist_ok=True)

    img_arr = cv2.imread(Path(source_dir_img) / img["file_name"])

    for ann in img_id_to_ann[img["id"]]:
        cv2.rectangle(img_arr, (ann["bbox"][COCO_BOX_XMIN_IDX], ann["bbox"][COCO_BOX_YMIN_IDX]), 
                      (ann["bbox"][COCO_BOX_XMIN_IDX] + ann["bbox"][COCO_BOX_WIDTH_IDX], ann["bbox"][COCO_BOX_YMIN_IDX] + ann["bbox"][COCO_BOX_HEIGHT_IDX]),
                      (0, 255, 0), 1)

    cv2.imwrite(str(output_path / f"full_img.jpg"))

    for key in patch_metadata_mapping_img:
        patch_dict = patch_metadata_mapping_img[key]
        patch_arr = img_arr[patch_dict["patch_y_min"] : patch_dict["patch_y_max"], patch_dict["patch_x_min"], patch_dict["patch_x_max"]]
        for box in patch_dict["boxes"]:
            x_min = box[YOLO_BOX_XCENTER_IDX] - box[YOLO_BOX_WIDTH_IDX] / 2
            xmax = box[YOLO_BOX_XCENTER_IDX] + box[YOLO_BOX_WIDTH_IDX] / 2
            ymin = box[YOLO_BOX_YCENTER_IDX] - box[YOLO_BOX_HEIGHT_IDX] / 2
            ymax = box[YOLO_BOX_YCENTER_IDX] + box[YOLO_BOX_HEIGHT_IDX] / 2


            cv2.rectangle(patch_arr, (x_min, ymin), (xmax, ymax), (0, 255, 0), 1)

            cv2.imwrite(str(output_path / f"{patch_dict['patch_name']}.jpg"))
