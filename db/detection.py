from db.base import BASE
import numpy as np

class DETECTION(BASE):
    def __init__(self, db_config):
        super().__init__()

        self._configs["categories"]      = 3
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2
        self._configs["border"]          = 128
        self._configs["input_size"]      = [511]
        self._configs["output_sizes"]    = [[128, 128]]
        self._configs["lighting"] = True
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 100

        self._configs["gaussian_bump"]   = True
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1
        self._configs["center_gaussian_radius"] = None

        self._configs["rand_color"] = False
        self._configs["rand_crop"] = False
        self._configs["annotation_mode"] = "default"
        self._configs["point_categories"] = []
        self._configs["line_center_mode"] = "nearest_real"
        self._configs["center_top_k"] = 50
        self._configs["key_nms_kernel"] = 1
        self._configs["center_nms_kernel"] = 3
        # Optional grouping inference thresholds. Unused for KPDetection unless set in config.
        self._configs["grouping_key_score_thresh"] = None
        self._configs["grouping_center_score_thresh"] = None
        self._configs["grouping_link_score_thresh"] = None
        self._configs["grouping_assignment_mode"] = "one_to_one"
        self._configs["grouping_min_points"] = 8
        self._configs["grouping_max_points"] = 120

        self.update_config(db_config)

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(
                self._configs["rand_scale_min"], 
                self._configs["rand_scale_max"],
                self._configs["rand_scale_step"]
            )
