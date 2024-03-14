import random 
import json
import shutil

from pathlib import Path
from tqdm import tqdm


def generate_splits(patches_mapping: dict, val_frac: float, test_frac: float, category_id_to_name: dict, 
                    base_dir: str, train_dir: str, val_dir: str, test_dir: str, radii: dict = None) -> dict:
    """
    Generate the training-, validation- and test-split from a dictionary mapping patches to metadata. 
    The method takes a path to the base directory, where all the patch annotation-files are stored, and moves them 
    to the correct folders according to the generated splits. 
    Arguments: 
        patches_mapping (dict):         dictionary mapping patch_ids to metadata (class_distribution, annotation
                                        coordinates, etc.)
        val_frac (float):               fraction of patches to be used for validation
        test_frac (float):              fraction of patches to be used for testing
        category_id_to_name (dict):     dictionary mapping category ids to categpry names
        base_dir (str):                 path to the base directory
        train_dir (str):                path to the training data directory
        val_dir (str):                  path to the validation data directory 
        test_dir (str):                 path to the test data directory
        radii (dict):                   dictionary containing the radii for each class. These values will be used as a
                                        distance-threshold (pixel-based Euclidean distance) beyond which detections 
                                        are considered false positive when running localization.
    Returns:
        Dictionary containing information (class distribution and size) about the splits
    """
    random.seed(0)

    patch_ids_list = list(patches_mapping.keys())
    
    n_val_patches_ids = int(val_frac*len(patch_ids_list))
    n_test_patches_ids = int(test_frac*len(patch_ids_list))

    # create splits
    random.shuffle(patch_ids_list)
    val_patch_ids = patch_ids_list[:n_val_patches_ids]
    test_patch_ids = patch_ids_list[n_val_patches_ids:(n_val_patches_ids + n_test_patches_ids)]
    train_patch_ids = patch_ids_list[(n_val_patches_ids + n_test_patches_ids):]


    
    with open(f"{train_dir}/train_mapping.json","w") as f:
        json.dump(train_patch_ids,f,indent=1)
        
    with open(f"{val_dir}/val_mapping.json","w") as f:
        json.dump(val_patch_ids,f,indent=1)

    with open(f"{test_dir}/test_mapping.json","w") as f:
        json.dump(test_patch_ids,f,indent=1)


    # Copy annotation files to train/val/test folders and collect split statistics
    train_distribution = {cat: 0 for cat in category_id_to_name.keys()}
    val_distribution = {cat: 0 for cat in category_id_to_name.keys()}
    test_distribution = {cat: 0 for cat in category_id_to_name.keys()}

    # For each patch
    for patch_name in tqdm(patches_mapping.keys(), total=len(patches_mapping)):
        patch_data = patches_mapping[patch_name]
        
        # Make sure we have an annotation file
        src_path_ann = f"{base_dir}/{patch_name}.txt"
        assert Path.exists(Path(src_path_ann))
        
        # Copy annotation file to the place it belongs and collect class distributions
        if patch_name in train_patch_ids:
            for class_id in train_distribution.keys():
                train_distribution[class_id] += patch_data["class_distribution"][class_id]
            target_folder = train_dir
        elif patch_name in val_patch_ids:
            for class_id in val_distribution.keys():
                val_distribution[class_id] += patch_data["class_distribution"][class_id]
            target_folder = val_dir
        elif patch_name in test_patch_ids:
            for class_id in test_distribution.keys():
                test_distribution[class_id] += patch_data["class_distribution"][class_id]
            target_folder = test_dir
        else:
            raise ValueError("Unassigned patch!")
        
        target_path_ann = f"{target_folder}/{Path(src_path_ann).name}"
        shutil.move(src_path_ann, target_path_ann)

    # make sure all annotation files have been moved (sanity check)
    assert not any (Path(base_dir).glob(".*"))

    #Generate the YOLO training dataset file
    with open(f"{base_dir}/dataset.yaml","w") as f:
        train_dir_rel = Path(train_dir).relative_to(base_dir)
        val_dir_rel = Path(val_dir).relative_to(base_dir)
        test_dir_rel = Path(test_dir).relative_to(base_dir)
        
        f.write("# Train/val/test sets\n" \
                f"path: {base_dir}\n" \
                f"train: {train_dir_rel}\n" \
                f"val: {val_dir_rel}\n" \
                f"test: {test_dir_rel}\n"\
                "\n" \
                "# Classes\n" \
                "names:\n")
        
        for class_id,class_name in category_id_to_name.items():
            f.write(f"  {class_id}: {class_name.strip()}\n")
        if radii: 
            f.write("\nradii:\n")
            for class_id, radius in radii.items():
                f.write(f"  {class_id}: {radius}\n")


    return {"train": {"distribution": train_distribution, "size": len(train_patch_ids)},
            "val": {"distribution": val_distribution, "size": len(val_patch_ids)},
            "test": {"distribution": test_distribution, "size": len(test_patch_ids)}}

