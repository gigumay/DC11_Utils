import shutil
from pathlib import Path


def retrieve_images(img_dir: str, ann_dir: str, copy: bool) -> int:
    """
    For each annotation file in the annotation directory, retrieve the 
    corresponding image from the image directory.
    Arguments:
        img_dir (str):          path to the directory containing the images
        ann_dir (str):          path to the directory containing the annotation files
        copy (bool):            if true, files are copied to the directory, else they are moved.
    Returns:
        the number of images retrieved
    """
    counter = 0
    for fn in Path(ann_dir).glob("*.txt"):
        img_fn = f"{img_dir}/{fn.stem}.jpg"
        if copy:
            shutil.copy(src=img_fn, dst=ann_dir)
        else: 
            shutil.move(src=img_fn, dst=ann_dir)
        counter += 1

    return counter


def relocate_images(src_dir: str, dest_dir: str) -> int:
    """
    Remove image files from the annotations directory and put them back into 
    he images directory.
    Arguments:
        src_dir (str):          path to the directory containing the images and annotations
        dest_dir (str):         path to the directory where the image files are stored
    Returns:
        the number of images relocated
    """
    counter = 0
    for fn in Path(src_dir).glob("*.jpg"):
        shutil.move(src=fn, dst=dest_dir)
        counter += 1

    return counter 


def load_imgs(img_dir: str, ann_dir: str, copy: bool) -> int:
    """
    Executes 'retrieve_imgs()' for a given annotation directory and performs a few
    sanity checks to ensure that no files were lost in case 'copy' is False. 
    Arguments:
        img_dir (str):          path to the directory containing the images
        ann_dir (str):          path to the directory containing the annotation files
        copy (bool):            if true, files are copied to the directory, else they are moved.
    Returns:
        the number of images retrieved
    """
    n_imgs = retrieve_images(img_dir=img_dir, ann_dir=ann_dir, copy=copy)

    assert n_imgs == len(list(Path((ann_dir)).glob("*.txt")))
    assert n_imgs == 0.5 * (len(list(Path((ann_dir)).glob("*.*"))) - 1)

    return n_imgs


def restore_dir(src_dir: str, dest_dir: str) -> int:
    """
    Executes 'relocate_imgs()' for a given annotation directory and performs a few
    sanity checks to ensure that no files were lost. 
    Arguments:
        img_dir (str):          path to the directory containing the images
        ann_dir (str):          path to the directory containing the annotation files
    Returns:
        the number of images relocated
    """
    n_files_dest_init = len(list(Path(dest_dir).glob("*.jpg")))
    n_imgs = relocate_images(src_dir=src_dir, dest_dir=dest_dir)

    assert len(list(Path(dest_dir).glob("*.jpg"))) == n_files_dest_init + n_imgs

    return n_imgs



def ready_yolo_dirs(img_dir: str, base_dir: str, copy: bool, subdirs: list = ["train", "val", "test"]) -> list:
    """
    Ready directories containing annotation files for training. I.e., load the corresponding images if mode == 'retrieve',
    and return the images to their original directory  . Probably not a very elegant solution.
    Arguments:
        img_dir (str):          path to the images directory
        base_dir (str):         path to the directory containing the sub-directories used during yolo training
                                (i.e., the training-, validation-, and test-directories)
        copy (bool):            if true, files are copied to the directory, not moved.
        subdirs (list):         names of the subdirectories
    Returns: 
        A list containing the number of retrieved images for each subdirectory
    """
    img_counts = []
    for dir in subdirs:
        img_counts.append(load_imgs(img_dir=img_dir, ann_dir=f"{base_dir}/{dir}", copy=copy))
    
    return img_counts



def clean_yolo_dirs(img_dir: str, base_dir: str, subdirs: list = ["train", "val", "test"]) -> list:
    """
    Remove the image files from the yolo directories.
    Arguments:
        img_dir (str):          path to the images directory
        base_dir (str):         path to the directory containing the sub-directories used during yolo training
                                (i.e., the training-, validation-, and test-directories)
        subdirs (list):         names of the subdirectories
    Returns: 
        A list containing the number of removed images for each subdirectory
    """
    img_counts = []
    for dir in subdirs:
        img_counts.append(restore_dir(src_dir=f"{base_dir}/{dir}", dest_dir=img_dir)) 

    return img_counts

