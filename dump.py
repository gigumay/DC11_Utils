def filter_ovrlp(coords: torch.Tensor, ovrlp_tresh: float = 0.9) -> torch.Tensor:

    coords_clone = torch.clone(coords)
    # get intersection between boxes
    (a1, a2), (b1, b2) = coords.unsqueeze(1).chunk(2, 2), coords_clone.unsqueeze(0).chunk(2, 2)

    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # get areas of boxes
    areas = (a2[:,:,0] - a1[:,:,0]) * (a2[:,:,1] - a1[:,:,1])

    # get overlap percentage
    ovrlp_pct = inter / areas

    excessive_ovrlps = torch.nonzero(ovrlp_pct >= ovrlp_tresh)
    pairs = excessive_ovrlps[torch.nonzero(excessive_ovrlps[:, 0] != excessive_ovrlps[:, 1])].squeeze(1)
    keep = torch.ones(pairs.shape[0]).long()
    for i in range(pairs.shape[0]):
        prior = pairs[:i, 1]
        if pairs[i, 0] in prior:
            keep[i] = 0

    pairs = pairs[torch.nonzero(keep)].squeeze(1)

    indices = pairs[:, 0]
    
    return indices

def greedy_nmm_bxs_SAHI(box_coords: torch.Tensor, scores: torch.Tensor, match_threshold: float):
    """
    Non-maximum merging as implemented in the SAHI framework.
    Arguments
        box_coords (torch.Tensor):                   coordinates of predicted bounding boxes
        scores (torch.Tensor):                       confidence scores of predictions.
        match_threshold (float):                     maximum allowed box overlap (IoU)
    Returns:
        keep_to_merge_list (Dict[int:List[int]]):    mapping from prediction indices to keep to a list 
                                                     of prediction indices to be merged.
    """
    keep_to_merge_list = {}

    # sort the prediction boxes in P according to their confidence scores
    order = scores.argsort()

    while len(order) > 0:
        # extract the index of the prediction with highest score we call this prediction S
        idx = order[-1]

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            keep_to_merge_list[idx.tolist()] = []
            break

        # select coordinates of points according to the indices in order
        bxs2compare = torch.index_select(box_coords, dim=0, index=order)


        # find the IoU of every prediction in P with S
        match_metric_value = bbox_iou(box1=box_coords[idx], box2=bxs2compare).squeeze(1)

        # keep the boxes with IoU greater than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))
        unmatched_indices = order[(mask == True).nonzero().flatten()]

        # update box pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        # create keep_ind to merge_ind_list mapping
        keep_to_merge_list[idx.tolist()] = []

        for matched_pt_ind in matched_box_indices.tolist():
            keep_to_merge_list[idx.tolist()].append(matched_pt_ind)

    return keep_to_merge_list



def greedy_nmm_pts_SAHI(pt_coords: torch.Tensor, scores: torch.Tensor, radii_t: torch.Tensor, match_threshold: float):
    """
    Non-maximum merging as implemented in the SAHI framework for points.
    Arguments
        pt_coords (torch.Tensor):                    coordinates of predicted points
        scores (torch.Tensor):                       confidence scores of predictions.
        match_threshold (float):                     maximum allowed proximity (DoR)
    Returns:
        keep_to_merge_list (Dict[int:List[int]]):    mapping from prediction indices to keep to a list 
                                                     of prediction indices to be merged.
    """
    keep_to_merge_list = {}

    # sort the prediction points in P according to their confidence scores
    order = scores.argsort()

    while len(order) > 0:
        # extract the index of the prediction with highest score we call this prediction S
        idx = order[-1]

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            keep_to_merge_list[idx.tolist()] = []
            break

        # select coordinates of points according to the indices in order
        pts2compare = torch.index_select(pt_coords, dim=0, index=order)


        # find the DoR of every prediction in P with S
        match_metric_value = loc_dor(loc1=pt_coords[idx], loc2=pts2compare, radii_t=radii_t[idx])

        # keep the points with DoR greater than thresh_iou
        mask = match_metric_value > match_threshold
        matched_pt_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))
        unmatched_indices = order[(mask == True).nonzero().flatten()]

        # update pt pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        # create keep_ind to merge_ind_list mapping
        keep_to_merge_list[idx.tolist()] = []

        for matched_pt_ind in matched_pt_indices.tolist():
            keep_to_merge_list[idx.tolist()].append(matched_pt_ind)

    return keep_to_merge_list




def run_tiled_inference_SAHI(model_file: str, task: str, class_ids: list, imgs_dir: str, img_files_ext: str, tiling_dir: str, 
                             patch_dims: dict, patch_overlap: float, vis_dir: str, det_dir: str, vis_prob: float, vis_density: int,
                             patch_quality: int = 95, radii: dict = None, save_pre_output: bool = False, iou_thresh: float = None, 
                             dor_thresh: float = None, rm_tiles: bool = True, save_patch_data: bool = False, 
                             verbose: bool = False) -> None:
   
    """
    Perform tiled inference on a directpry of images, but using the SAHI stitching mechanism.
    Arguments:
        model_file (str):           path to the model file (.pt) to be used for inference.
        task (str):                 task string. Can be 'detect' or 'locate'.
        class_ids (list):           list of integer class IDs in the data the model was trained on.
        imgs_dir (str):             path to the folder containing the images inference needs to be performed on.
        img_files_ext (str):        Extension of image files. E.g., 'JPG'.
        tiling_dir (str):           path to the folder where tiles can be stored temporarily.
        patch_dims (dict):          dictionary with keys 'width' & 'height' specifying the tile dimensions. 
        patch_overlap (float):      amount of overlap (fraction) between tiles. 
        vis_dir (str):              path to the folder where plots of images containing predictions can be stored.
        det_dir (str):              path to the folder where inference outputs can be stored (.json).
        vis_prob (str):             plotting probability. For each image, a random number between 0 and 1 is drawn, 
                                    and if it's below vis_prob, the image in question & the corresponding predictions
                                    will be visualized. 
        vis_density (int):          number of objects/animals that will trigger plotting. If the number of predictions 
                                    for an image is equal to or exceeds vis_prob, the image & predictions will be plotted
                                    regardless of vis_prob.
        patch_quality (float):      quality of tle image files (not sure what this does or why I need it tbh).
        radii (dict):               dictionary containing the radius values for localization NMS.
        save_pre_output (bool):     if True, outputs (plots + detections) before NMS will be sav ed as well.
        iou_thresh (float):         IoU threshold to use during NMS when stitching tiles together. 
        dor_thresh (float):         DoR threshold to use during NMS when stitching tiles together. 
        rm_tiles (bool):            whether to remove the tile image files after inference.
        save_patch_data (bool):     if True, the predictions for each tile will bes tored as files too (.json)
        verbose (bool):             if True, the amount of predictions removed by NMS will be printed to the console.
    Returns:
        None
    """

    assert patch_overlap < 1 and patch_overlap >= 0, \
        'Illegal tile overlap value {}'.format(patch_overlap)
    
    model = YOLO(model_file)
    
    img_fns = list(Path(imgs_dir).rglob(f"*.{img_files_ext}"))
    counts_sum = {cls_id: 0 for cls_id in class_ids}
    
    print("*** Processing images")
    for fn in tqdm(img_fns, total=len(img_fns)):
        im = vis_utils.open_image(fn)
                
        patch_start_positions = get_patch_start_positions(img_width=im.width, img_height=im.height, patch_dims=patch_dims, 
                                                          overlap=patch_overlap) 
        
        patches = []
        for patch in patch_start_positions: 
            patch_coords = {"x_min": patch[0], 
                            "y_min": patch[1], 
                            "x_max": patch[0] + patch_dims["width"] - 1,
                            "y_max": patch[1] + patch_dims["height"] - 1}
            
            patch_name = patch_info2name(image_name=fn.name, patch_x_min=patch_coords['x_min'], patch_y_min=patch_coords['y_min'])
            patch_fn = f"{tiling_dir}/{patch_name}.jpg"
            
            patch_metadata = {"patch_fn": patch_fn,
                              "patch_name": patch_name,
                              "coords": patch_coords}
            
            patches.append(patch_metadata)
        
            patch_im = im.crop((patch_coords["x_min"], patch_coords["y_min"], patch_coords["x_max"] + 1,
                                patch_coords["y_max"] + 1))
            assert patch_im.size[0] == patch_dims["width"]
            assert patch_im.size[1] == patch_dims["height"]
        
            assert not Path(patch_fn).exists()
            patch_im.save(patch_fn, quality=patch_quality)

    
        # create folder to store original patch level predictions
        if save_patch_data:
            patch_data_dir = str(Path(det_dir).parent / "patch_data")
            patch_data_dir.mkdir(parents=False, exist_ok=True)
            patch_output_dir = f"{patch_data_dir}/{fn.stem}"
            Path(patch_output_dir).mkdir(parents=False, exist_ok=True)
        else:
            patch_output_dir = None

        # run detection on patches 
        patch_fns = [patch["patch_fn"] for patch in patches]
        if task == "detect":
            predictions = model(input_arrays, verbose=False)
        else:
            predictions = model(patch_fns, radii=radii, verbose=False)
        
        # collect predictions from each patch and map it back to image level
        coords, conf, cls = collect_predictions_wrapper(task=task, predictions=predictions, patches=patches, 
                                                        device=model.device, patch_output_dir=patch_output_dir)
        
        if task == 'locate':
            radii_t = generate_radii_t(radii=radii, cls=cls_idx_pre)
        
        # get counts before nms
        pre_nms = coords.shape[0]
        cls_idx_pre, counts_pre = torch.unique(cls.squeeze(1), return_counts=True)
        counts_pre_dict = {}
        for j in range(cls_idx_pre.shape[0]):
            counts_pre_dict[int(cls_idx_pre[j].item())] = int(counts_pre[j].item())

        if save_pre_output:
            with open(f"{det_dir}/{fn.stem}_pre_nms.json", "w") as f:
                json.dump(counts_pre_dict, f, indent=1)

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or pre_nms >= vis_density

        if visualize and save_pre_output:
            plot_annotated_img(img_fn=str(fn), coords=coords, pre_nms=True, output_dir=vis_dir)

        # starting from the most confident prediction, greedily find predictions that exceed IoU/DoR, and merge
        keep_to_merge_list = {}
        for category_id in cls_idx_pre:
            curr_indices = torch.where(cls == category_id)[0]
            if task == 'detect':
                curr_keep_to_merge_list = greedy_nmm_bxs_SAHI(box_coords=coords[curr_indices], 
                                                              scores=conf[curr_indices].squeeze(1), 
                                                              match_threshold=iou_thresh)
            else:
                curr_keep_to_merge_list = greedy_nmm_pts_SAHI(pt_coords=coords[curr_indices], 
                                                              scores=conf[curr_indices].squeeze(1), 
                                                              radii_t=radii_t[curr_indices], 
                                                              match_threshold=dor_thresh)
            curr_indices_list = curr_indices.tolist()
            for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
                keep = curr_indices_list[curr_keep]
                merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
                keep_to_merge_list[keep] = merge_list

        idxs = torch.IntTensor(list(keep_to_merge_list.keys()))

        # combine coordinates, confidence and class into one tensor
        preds_img_final = torch.hstack((coords[idxs], conf[idxs], cls[idxs]))

        # get counts post nms 
        post_nms = preds_img_final.shape[0]
        cls_idx_post, counts_post = torch.unique(preds_img_final[:, -1], return_counts=True)
        counts_post_dict = {}
        for j in range(cls_idx_post.shape[0]):
            counts_post_dict[int(cls_idx_post[j].item())] = int(counts_post[j].item())

        with open(f"{det_dir}/{fn.stem}.json", "w") as f:
            json.dump(counts_post_dict, f, indent=1)
        
        if verbose:
            print(f"NMS removed {pre_nms - post_nms} predictions!")

        # add to count sum
        for class_idx, n in counts_post_dict.items():
            counts_sum[class_idx] += n

        if visualize:
            plot_annotated_img(img_fn=str(fn), coords=coords[idxs], pre_nms=False, output_dir=vis_dir)

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)

    if rm_tiles:
        shutil.rmtree(tiling_dir) 




def get_total_counts_SAHI(sahi_results: str, class_ids: list) -> None:
    """
    Collect total counts from SAHI tiled inference output.
    Arguments:
        sahi_results (str):       path to the sahi result file
        class_ids (list):         list of integer class ids. 
    Returns:
        None
    """

    with open(sahi_results, "r") as f:
        pred_dict = json.load(f)

    total_counts = {cls_id: 0 for cls_id in class_ids}

    for pred in pred_dict:
        total_counts[pred["category_id"]] += 1
    
    output_fn = f"{Path(sahi_results).parent}/counts_total.json"
    with open(output_fn, "w") as f:
        json.dump(total_counts, f, indent=1)


def compute_errors_patch_lvl(patch_data_dir: str, ann_file: str, task: str, class_ids: list, data_ann_format: str, output_dir: str,
                             box_dims: dict = None) -> dict:
    """
    Given a directory containing the patch predictions for an image set, compute the patch level MAE and MSE
    """
    assert task in ["detect", "locate"]
    
    counter = 0
    mae_overall = {cls_id: 0 for cls_id in class_ids}
    mae_overall["overall"] = 0
    mse_overall = {cls_id: 0 for cls_id in class_ids}
    mse_overall["overall"] = 0    

    with open(ann_file, "r") as f:
        all_anns = json.load(f)

    for child in Path(patch_data_dir).iterdir():
        if child.is_dir():
            fn = child.stem
            key = [img_name for img_name in all_anns.keys() if fn in img_name]
            assert len(key) == 1
            ann_dict = all_anns[key[0]]

            for patch_file in Path(child).iterdir():
                patch_str = patch_file.stem
                xmin_patch = int(patch_str.split('_')[0])
                ymin_patch = int(patch_str.split('_')[1])

                with open(patch_file, "r") as f:
                    patch_counts = json.load(f)

                gt_patch = {cls_id: 0 for cls_id in class_ids}

                for ann in ann_dict:
                    label = ann[DATA_ANN_FORMATS[data_ann_format]["label_key"]]
                    cat = ann[DATA_ANN_FORMATS[data_ann_format]["category_key"]]
                    if data_ann_format == "BX_WH":
                        x_center = label[DATA_ANN_FORMATS[data_ann_format]["x_min_idx"]] + \
                                   (label[DATA_ANN_FORMATS[data_ann_format]["width_idx"]] / 2.0)
                        y_center = label[DATA_ANN_FORMATS[data_ann_format]["y_min_idx"]] + \
                                   (label[DATA_ANN_FORMATS[data_ann_format]["height_idx"]] / 2.0)

                        box_dims_checked = {}

                        if box_dims:
                            box_dims_checked["width"] = box_dims["width"]
                            box_dims_checked["height"] = box_dims["height"]
                        else: 
                            box_dims_checked["width"] = label[DATA_ANN_FORMATS[data_ann_format]["width_idx"]]
                            box_dims_checked["height"] = label[DATA_ANN_FORMATS[data_ann_format]["height_idx"]]

                        xmax = x_center + (box_dims_checked["width"]/2.0)
                        ymax = y_center + (box_dims_checked["height"]/2.0)
                        if task == "detect":
                            within_patch = xmax > xmin_patch and ymax > ymin_patch
                        if task == "locate":
                            within_patch = x_center >= xmin_patch and y_center >= ymin_patch


                    if data_ann_format == "PT_DEFAULT":
                        x_ann= label["x_idx"]
                        y_ann = label["y_idx"]
                        within_patch = x_ann >= xmin_patch and y_ann >= ymin_patch

                    if within_patch:
                        gt_patch[cat] += 1

                abs_err = {cls_id: abs(gt_patch[cls_id] - patch_counts[str(cls_id)]) for cls_id in class_ids}
                overall = abs(sum(gt_patch.values()) - sum(patch_counts.values()))
                for cls_id in class_ids:
                    mae_overall[cls_id] += abs_err[cls_id]
                    mse_overall[cls_id] += abs_err[cls_id] **2
                mae_overall["overall"] += overall
                mse_overall["overall"] += overall **2

                counter += 1

    results = {cls_id: {"MAE": mae_overall[cls_id] / counter, "MSE": mse_overall[cls_id] / counter} for cls_id in class_ids}
    results["overall"] = {"MAE": mae_overall["overall"] / counter, "MSE": mse_overall["overall"] / counter}
    
    with open(f"{output_dir}/errors_patch_lvl.json", "w") as f:
        json.dump(results, f, indent=1)

    return results
