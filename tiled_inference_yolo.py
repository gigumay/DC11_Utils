import sys
sys.path.append("/home/giacomo/repos/MegaDetector")

import tqdm
import json
import visualization_utils as visutils

from pathlib import Path
from ultralytics import YOLO

from processing_utils import *
from globals import *

def run_tiled_inference(model_file: str, imgs_dir: str, tiling_dir: str, patch_dims: dict, patch_overlap: float, 
                        patch_quality: int, meta_results_dir: str, vis_dir: str, det_dir: str, vis_prob: float,
                        task: str, iou_thresh: float = None, dor_thresh: float = None) -> None:
    """
    Adjusted based on MegaDetector code!

    Runs inference using [model_file] on the images in [image_folder], fist splitting each image up 
    into tiles of size [tile_size_x] x [tile_size_y], writing those tiles to [tiling_folder],
    then de-duplicating the results before merging them back into a set of detections that make 
    sense on the original images and writing those results to [output_file].  
    
    [tiling_folder] can be any folder, but this function reserves the right to do whatever it wants
    within that folder, including deleting everything, so it's best if it's a new folder.  
    Conceptually this folder is temporary, it's just helpful in this case to not actually
    use the system temp folder, because the tile cache may be very large, so the caller may 
    want it to be on a specific drive.
    
    tile_overlap is the fraction of overlap between tiles.
    
    Optionally removes the temporary tiles.
    
    if yolo_inference_options is supplied, it should be an instance of YoloInferenceOptions; in 
    this case the model will be run with run_inference_with_yolov5_val.  This is typically used to 
    run the model with test-time augmentation.
    
    Args:
        model_file (str): model filename (ending in .pt), or a well-known model name (e.g. "MDV5A")
        image_folder (str): the folder of images to proess (always recursive)
        tiling_folder (str): folder for temporary tile storage; see caveats above
        output_file (str): .json file to which we should write MD-formatted results
        tile_size_x (int, optional): tile width
        tile_size_y (int, optional): tile height
        tile_overlap (float, optional): overlap between adjacenet tiles, as a fraction of the
            tile size
        checkpoint_path (str, optional): checkpoint path; passed directly to run_detector_batch; see
            run_detector_batch for details
        checkpoint_frequency (int, optional): checkpoint frequency; passed directly to run_detector_batch; see
            run_detector_batch for details
        remove_tiles (bool, optional): whether to delete the tiles when we're done
        yolo_inference_options (YoloInferenceOptions, optional): if not None, will run inference with
            run_inference_with_yolov5_val.py, rather than with run_detector_batch.py, using these options
        n_patch_extraction_workers (int, optional): number of workers to use for patch extraction;
            set to <= 1 to disable parallelization
        image_list (list, optional): .json file containing a list of specific images to process.  If 
            this is supplied, and the paths are absolute, [image_folder] will be ignored. If this is supplied,
            and the paths are relative, they should be relative to [image_folder].
    
    Returns:
        dict: MD-formatted results dictionary, identical to what's written to [output_file]
    """

    assert patch_overlap < 1 and patch_overlap >= 0, \
        'Illegal tile overlap value {}'.format(patch_overlap)
    
    imgs_iterator = Path(imgs_dir).rglob("*.jpg")
    
    print("*** Processing images")
    for fn in tqdm(imgs_iterator, total=len(list(imgs_iterator))):
        im = vis_utils.open_image(str(fn))
                
        patch_start_positions = get_patch_start_positions(img_width=im.width, img_height=im.height, patch_dims=patch_dims, 
                                                          overlap=patch_overlap) 
        
        patches = []
        for patch in patch_start_positions: 
            patch_coords = {"x_min": patch[PATCH_XSTART_IDX], 
                            "y_min": patch[PATCH_YSTART_IDX], 
                            "x_max": patch[PATCH_XSTART_IDX] + patch_dims["width"] - 1,
                            "y_max": patch[PATCH_YSTART_IDX] + patch_dims["height"] - 1}
            
            patch_name = patch_info2name(image_name=fn.name, patch_x_min=patch_coords['x_min'], patch_y_min=patch_coords['y_min'])
            patch_fn = f"{tiling_dir}/{patch_name}.jpg"
            
            patch_metadata = {"patch_fn": patch_fn,
                              "patch_name": patch_name,
                              "patch_x_min": patch_coords["x_min"],
                              "patch_y_min": patch_coords["y_min"],
                              "patch_x_max": patch_coords["x_max"],
                              "patch_y_max": patch_coords["y_max"]}
            
            patches.append(patch_metadata)
        
            patch_im = im.crop((patch_coords["x_min"], patch_coords["y_min"], patch_coords["x_max"] + 1,
                                patch_coords["y_max"] + 1))
            assert patch_im.size[0] == patch_dims["width"]
            assert patch_im.size[1] == patch_dims["height"]
        
            assert not Path(patch_fn).exists()
            patch_im.save(patch_fn, quality=patch_quality)

    

        # run detection on patches 
        patch_fns = [patch["patch_fn"] for patch in patches]
        model = YOLO(model_file)
        detections = model(patch_fns)
        # write predictions to per patch 
        # map to image 
        # write to file
        # perform nms 
        # write to file again 
        # visualize
        patch_predictions_dir = Path(f"{meta_results_dir}/{fn.stem}").mkdir(parents=False, exist_ok=True)

            
