import sys
sys.path.append("/home/giacomo/repos/MegaDetector/md_visualization")

import math
import cv2

from pathlib import Path

import visualization_utils as visutils


YOLO_BOX_CAT_IDX = 0
YOLO_BOX_XCENTER_IDX = 1
YOLO_BOX_YCENTER_IDX = 2
YOLO_BOX_WIDTH_IDX = 3
YOLO_BOX_HEIGHT_IDX = 4

YOLO_PT_CAT_IDX = 0
YOLO_PT_X_IDX = 1
YOLO_PT_Y_IDX = 2

COCO_BOX_XMIN_IDX = 0
COCO_BOX_YMIN_IDX = 1
COCO_BOX_WIDTH_IDX = 2
COCO_BOX_HEIGHT_IDX = 3

INPUT_FORMAT_Y_IDX = 0
INPUT_FORMAT_X_IDX = 1

PATCH_XSTART_IDX = 0
PATCH_YSTART_IDX = 1

MAX_OVERHANG_BXS = 0.85
MAX_OVERHANG_PTS = 0.45
PT_VIS_CIRCLE_RADIUS = 2



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
            if overhang > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["width"] -= overhang
            x_center_relative -= overhang / 2.0

        box_bottom = y_center_relative + (box_dims_rel["height"] / 2.0)                                        
        if box_bottom > 1.0:
            overhang = box_bottom - 1.0
            if overhang > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["height"] -= overhang
            y_center_relative -= overhang / 2.0
        
        box_left = x_center_relative - (box_dims_rel["width"] / 2.0)
        if box_left < 0.0:
            overhang = abs(box_left)
            if overhang > max_overhang:
                continue
            clipped_box = True
            box_dims_rel["width"] -= overhang
            x_center_relative += overhang / 2.0
            
        box_top = y_center_relative - (box_dims_rel["height"] / 2.0)
        if box_top < 0.0:
            overhang = abs(box_top)
            if overhang > max_overhang:
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

    class_distr_patch = {cat: 0 for cat in categories}
    gt_points = []
            
    assert (1 + patch_coords["x_max"] - patch_coords["x_min"]) == patch_dims["width"]
    assert (1 + patch_coords["y_max"] - patch_coords["y_min"]) == patch_dims["height"]

    for ann in annotations_in_img:                
        gt_x = ann["x"]
        gt_y = ann["y"]
        
        patch_contains_pt = (patch_coords["x_min"] < gt_x and gt_x < patch_coords["x_max"] \
                             and patch_coords["y_min"] < gt_y and gt_y < patch_coords["y_max"])
        
        if patch_contains_pt:
            x_coords_absolute_patch = gt_x - patch_coords["x_min"]
            y_coords_absolute_patch = gt_y - patch_coords["y_min"]

            #Again, relative coordinates 
            x_coords_relative_patch = x_coords_absolute_patch / patch_dims["width"]
            y_coords_relative_patch = y_coords_absolute_patch / patch_dims["height"]
            
            gt_points.append([ann["category_id"], x_coords_relative_patch, y_coords_relative_patch])
            class_distr_patch[ann['category_id']] += 1
    
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
            '''
            When dealing with points, boxes cannot be clipped and thus  much larger parts of 
            animals that cross patch borders will be excluded and treated as background. To simulate this,
            I use yolo boxes filtered with a stricter overhang parameter to retrieve the point annotations.
            '''
            yolo_boxes, class_distr, _ = get_boxes_at_patch_lvl(annotations_in_img=annotations_in_img,
                                                                patch_dims=patch_dims, patch_coords=patch_coords, 
                                                                categories=categories, max_overhang=MAX_OVERHANG_PTS,
                                                                box_dims=box_dims)
            gt_points = [[box[YOLO_BOX_CAT_IDX], box[YOLO_BOX_XCENTER_IDX], box[YOLO_BOX_YCENTER_IDX]] for box in yolo_boxes]
            return gt_points, class_distr, 0
        
        
    else: 
         gt_points, class_distr = get_points_in_patch(annotations_in_img=annotations_in_img, patch_dims=patch_dims, 
                                                      patch_coords=patch_coords, categories=categories)
         return gt_points, class_distr, 0
        
    


def process_image(source_dir_img: str, img: dict, img_width: int, img_height: int, boxes_in: bool, boxes_out: bool, 
                  img_id_to_ann: dict, patch_dims: dict, patch_start_positions: list, patch_jpeg_quality: int,
                  categories: list, visualize: bool, dest_dir_imgs: tuple[None, str], dest_dir_txt: str, 
                  box_dims: dict = None, vis_output_dir: str = None) -> dict:
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
        patch_jpeg_quality (int):       quality of the patch-images
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
    class_distr_img = {cat: 0 for cat in categories}

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
        
        # skip empty patches in positive images
        if not gt:
            continue 

        n_boxes_clipped_img += n_boxes_clipped
        n_annotations_img += len(gt)
        class_distr_img = {**class_distr_img, **patch_distr}

        
        patch_name = patch_info_to_patch_name(img["id"], patch_coords["x_min"], patch_coords["y_min"])
        patch_ann_file = Path(dest_dir_txt) / f"{patch_name}.txt"
        
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
           "class_distribution": patch_distr
        }

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
            

            patch_image_file = Path(dest_dir_imgs) / f"{patch_name}.jpg"
            assert not patch_image_file.exists()
            patch_im.save(patch_image_file, quality=patch_jpeg_quality)
        
        with open(patch_ann_file, 'w') as f:
            for ann in gt:
                if boxes_out:
                    ann_str = f"{ann[YOLO_BOX_CAT_IDX]} {ann[YOLO_BOX_XCENTER_IDX]} " \
                                f"{ann[YOLO_BOX_YCENTER_IDX]} {ann[YOLO_BOX_WIDTH_IDX]} " \
                                f"{ann[YOLO_BOX_HEIGHT_IDX]}\n"
                else:
                    ann_str = f"{ann[YOLO_PT_CAT_IDX]} {ann[YOLO_PT_X_IDX]} {ann[YOLO_PT_Y_IDX]}\n"
                
                f.write(ann_str)
                
        assert Path.exists(patch_ann_file)

        n_patches_img += 1
    
    if visualize:
        assert vis_output_dir, "Please provide a path to a directory where visulaizations can be stored!"
        vis_processed_img(img=img, source_dir_img=source_dir_img, img_id_to_ann=img_id_to_ann, patch_metadata_mapping_img=patch_metadata_mapping_img,
                          patch_dims=patch_dims, boxes_out=boxes_out, output_dir=vis_output_dir)

    return {"patches_mapping": patch_metadata_mapping_img, "n_patches": n_patches_img,
            "n_annotations": n_annotations_img, "n_boxes_clipped":  n_boxes_clipped_img, 
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


#TODO: Add functionality to add empty patches so that negative sampling doesn't rely on images beigng annotated as empty
