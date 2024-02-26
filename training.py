import shutil
from pathlib import Path


def retrieve_images(img_dir: str, ann_dir: str) -> int:
    """
    For each annotation file in the annotation directory, retrieve the 
    corresponding image from the image directory.
    Arguments:
        img_dir (str):          path to the directory containing the images
        ann_dir (str):          path to the directory containing the annotation files
    Returns:
        the number of images retrieved
    """
    counter = 0
    for fn in Path(ann_dir).glob("*.txt"):
        img_fn = f"{img_dir}/{fn.stem}.jpg"
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


def load_imgs(img_dir: str, ann_dir: str) -> int:
    """
    Executes 'retrieve_imgs()' for a given annotation directory and performs a few
    sanity checks to ensure that no files were lost. 
    Arguments:
        img_dir (str):          path to the directory containing the images
        ann_dir (str):          path to the directory containing the annotation files
    Returns:
        the number of images retrieved
    """
    n_imgs = retrieve_images(img_dir=img_dir, ann_dir=ann_dir)

    assert n_imgs == len(list(Path((ann_dir)).glob("*.txt")))
    assert n_imgs == 0.5 * len(list(Path((ann_dir)).glob("*.*")))

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



def ready_yolo_dirs(img_dir: str, base_dir: str, mode: str, subdirs: list = ["train", "val", "test"]) -> list:
    """
    Ready directories containing annotation files for training. I.e., laod the corresponding images (if mode == 'retrieve'),
    and return the images to their original directory. Probably not a very elegant solution.
    Arguments:
        img_dir (str):          path to the images directory
        base_dir (str):         path to the directory containing the sub-directories used during yolo training
                                (i.e., the training-, validation-, and test-directories)
        mode (str):             whether to load the images into the respective directories, or put them back into 
                                the original images directory
        subdirs (list):         names of the subdirectories
    Returns: 
        A list containing the number of retrieved/relocated images for each subdirectory
    """
    img_counts = []
    for dir in subdirs:
        if mode == "retrieve":
            img_counts.append(retrieve_images(img_dir=img_dir, ann_dir=f"{base_dir}/{dir}"))
        elif mode == "relocate":
            img_counts.append(relocate_images(src_dir=f"{base_dir}/{dir}", dest_dir=img_dir)) 
        else:
            raise ValueError("Invalid mode passed to directory prep!")
    
    return img_counts

