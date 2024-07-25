import cv2

from pathlib import Path
from tqdm import tqdm
from pathlib import Path

from globs import *

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
   
    patch_stride = {"x": round(patch_dims["width"]*(1.0-overlap)), "y": round(patch_dims["height"]*(1.0-overlap))}

    patch_width = patch_dims["width"]
    patch_height = patch_dims["height"]

    if patch_width >= img_width:
        print(f"WARNING: patch width ({patch_width}) is bigger than image width ({img_width}).\n" \
              f"Setting image width as pacth width!")
        patch_width = img_width
    
    if patch_height >= img_height:
        print(f"WARNING: patch width ({patch_height}) is bigger than image width ({img_height}).\n" \
              f"Setting image width as pacth width!")
        patch_height = img_height

    
    def add_patch_row(patch_start_positions,y_start):
        """
        Add one row to the list of patch start positions, i.e. loop over all columns.
        """
        x_start = 0; x_end = x_start + patch_width - 1
        
        while(True):
            patch_start_positions.append([x_start,y_start])
            
            if x_end == img_width - 1:
                break
            
            # Move one patch to the right
            x_start += patch_stride["x"]
            x_end = x_start + patch_width - 1
             
            # If this patch flows over the edge, add one more patch to cover the pixels at the end
            if x_end > (img_width - 1):
                overshoot = (x_end - img_width) + 1
                x_start -= overshoot
                x_end = x_start + patch_width - 1
                patch_start_positions.append([x_start,y_start])
                break
            
        return patch_start_positions
        
    patch_start_positions = []
    
    y_start = 0; y_end = y_start + patch_height - 1
    
    while(True):
        patch_start_positions = add_patch_row(patch_start_positions,y_start)
        
        if y_end == img_height - 1:
            break
        
        # Move one patch down
        y_start += patch_stride["y"]
        y_end = y_start + patch_height - 1
        
        # If this patch flows over the bottom, add one more patch to cover the pixels at the bottom
        if y_end > (img_height - 1):
            overshoot = (y_end - img_height) + 1
            y_start -= overshoot
            y_end = y_start + patch_height - 1
            patch_start_positions = add_patch_row(patch_start_positions,y_start)
            break
    
    for p in patch_start_positions:
        assert p[0] >= 0 and p[1] >= 0 and p[0] <= img_width and p[1] <= img_height, \
        f"Patch generation error (illegal patch {p})!"
        
    # The last patch should always end at the bottom-right of the image
    assert patch_start_positions[-1][0] + patch_width == img_width, \
        "Patch generation error (last patch does not end on the right)"
    assert patch_start_positions[-1][1] + patch_height == img_height, \
        "Patch generation error (last patch does not end at the bottom)"
    
    # All patches should be unique
    patch_start_positions_tuples = [tuple(x) for x in patch_start_positions]
    assert len(patch_start_positions_tuples) == len(set(patch_start_positions_tuples)), \
        "Patch generation error (duplicate start position)"
    
    return patch_start_positions

    

def patch_info2name(image_name: str, patch_x_min: int, patch_y_min: int, is_empty: bool = None) -> str:
    """
    Generates the name of a patch.
    Arguments: 
        image_name (str):   name of the image that contains the patch
        patch_x_min (int):  x coordinate of the left upper corner of the patch within the 
                            original image
        patch_y_min (int):  y coordinate of the left upper corner of the patch within the 
                            original image 
        is_empty (bool):    indicates whether the patch is empty
    Returns: 
        string containing the patch name
    """
    empty_ext = "_empty" if is_empty else ""
    patch_name = f"{image_name}_{str(patch_x_min).zfill(4)}_{str(patch_y_min).zfill(4)}{empty_ext}"
    return patch_name



def vis_img(img_path: str, annotations: list, ann_format: str, boxes_in: bool, boxes_out: bool, output_dir: str) -> None:
    """
    Plot an image with its labels.
    Arguments:
        img_path (str):             path to the image to be plotted
        annotations (list):         list of labels wihtin the image
        ann_format (str):           format in which the annotations are provided
        boxes_in (bool):            if true, annotations are expected to be box labels
        boxes_out (bool):           if true, annotations are plotted as boxes. Requires boxes_in to be true. 
    Returns:
        None
    """
    img_arr = cv2.imread(img_path)
    img_name = Path(img_path).stem

    for ann in annotations:
        label = ann[DATA_ANN_FORMATS[ann_format]["label_key"]]
        cls_id = ann[DATA_ANN_FORMATS[ann_format]["category_key"]]
        
        if boxes_out:
            assert boxes_in, "Cannot plot boxes from point labels"
            #cv2 needs int coordinates
            xmin_img = round(label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]])
            xmax_img = round(label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["width_idx"]])
            ymin_img = round(label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]])
            ymax_img = round(label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["height_idx"]])
         
            cv2.rectangle(img=img_arr, pt1=(xmin_img, ymin_img), pt2=(xmax_img, ymax_img), 
                          color=CLASS_COLORS[cls_id], thickness=1)
        else:
            if boxes_in:
                x_center = round(label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] +label[DATA_ANN_FORMATS[ann_format]["width_idx"]] / 2.0)
                y_center = round(label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["height_idx"]] / 2.0)
            else:
                x_center = label[DATA_ANN_FORMATS[ann_format]["x_idx"]]
                y_center = label[DATA_ANN_FORMATS[ann_format]["y_idx"]]
            
            cv2.circle(img=img_arr, center=(x_center, y_center), radius=3, color=CLASS_COLORS[cls_id], thickness=-1)

    cv2.imwrite(f"{output_dir}/{img_name}_ann.jpg", img_arr)

    