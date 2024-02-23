import random 
import json
import shutil

from pathlib import Path
from tqdm import tqdm

def generate_splits(patches_mapping: dict, val_frac: float, test_frac: float, train_imgs_fn: str, 
                    val_imgs_fn: str, test_imgs_fn: str, category_id_to_name: dict, base_dir: str,
                    train_dir: str, val_dir: str, test_dir: str):
    
    patch_ids_list = list(patches_mapping.keys())
    
    n_val_image_ids = int(val_frac*len(patch_ids_list))
    n_test_image_ids = int(test_frac*len(patch_ids_list))

    # write out train-/val-/test-set
    random.shuffle(patch_ids_list)
    val_image_ids = patch_ids_list[:n_val_image_ids]
    test_image_ids = patch_ids_list[n_val_image_ids:(n_val_image_ids + n_test_image_ids)]
    train_image_ids = patch_ids_list[(n_val_image_ids + n_test_image_ids):]


    
    with open(train_imgs_fn,'w') as f:
        json.dump(train_image_ids,f,indent=1)
        
    with open(val_imgs_fn,'w') as f:
        json.dump(val_image_ids,f,indent=1)

    with open(test_imgs_fn,'w') as f:
        json.dump(test_image_ids,f,indent=1)


    # Copy images to train/val folders and collect split statistics
    train_patch_names = []
    train_patch_classes = {cat: 0 for cat in category_id_to_name.keys()}
    val_patch_names = []
    val_patch_classes = {cat: 0 for cat in category_id_to_name.keys()}
    test_patch_names = []
    test_patch_classes = {cat: 0 for cat in category_id_to_name.keys()}

    # For each patch
    for patch_name in tqdm(patches_mapping.keys(), total=len(patches_mapping)):
        patch_data = patches_mapping[patch_name]
        
        # Make sure we have an annotation file
        src_path_ann = f"{base_dir}/{patch_name}.txt"
        assert Path.exists(src_path_ann)
        
        # Copy to the place it belongs and collect class distributions
        if patch_data['original_image_id'] in train_image_ids:
            train_patch_names.append(patch_name)
            train_patch_classes = {**train_patch_classes, **patch_data["class_distribution"]}
            target_folder = train_dir
        elif patch_data['original_image_id'] in val_image_ids:
            val_patch_names.append(patch_name)
            val_patch_classes = {**val_patch_classes, **patch_data["class_distribution"]}
            target_folder = val_dir
        elif patch_data['original_image_id'] in test_image_ids:
            test_patch_names.append(patch_name)
            test_patch_classes = {**test_patch_classes, **patch_data["class_distribution"]}
            target_folder = test_dir
        
        target_path_ann = f"{target_folder}/{Path(src_path_ann).name}"
        shutil.move(src_path_ann, target_path_ann)

    
    assert not any (Path(base_dir).glob(".*"))
    print(f"\nMoved {len(train_patch_names)} train patches, {len(val_patch_names)} val patches, {len(test_patch_names)} test patches")

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

