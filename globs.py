DATA_ANN_FORMATS = {    
    "BX_WH": {"label_key": "bbox", "x_min_idx": 0, "y_min_idx": 1, "width_idx": 2, "height_idx": 3, "category_key": "category_id"},
    "PT_DEFAULT": {"label_key": "point", "x_idx": 0, "y_idx": 1, "category_key": "category_id"}
}


MODEL_ANN_FORMATS = {
    "YOLO_BOX": {"center_x_idx": 1, "center_y_idx": 2, "width_idx": 3, "height_idx": 4, "category_idx": 0},
    "YOLO_PT":  {"x_idx": 1, "y_idx": 2, "category_idx": 0}
}

PATCH_XSTART_IDX = 0
PATCH_YSTART_IDX = 1

# BGR colors
CLASS_COLORS = {0: (134, 22, 171),      # purple
                1: (0, 255, 255),       # yellow
                2: (204, 232, 19),      # turquoise
                3: (0, 97, 242),        # orange
                4: (0, 255, 0)}         # green