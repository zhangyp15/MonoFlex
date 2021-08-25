import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""
_C.MODEL.PRETRAIN = True
_C.MODEL.USE_SYNC_BN = False
_C.MODEL.REDUCE_LOSS_NORM = True
_C.MODEL.NORM = 'BN' # group normalization or batch normalization

_C.MODEL.INPLACE_ABN = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.HEIGHT_TRAIN = 384
# Maximum size of the side of the image during training
_C.INPUT.WIDTH_TRAIN = 1280
# Size of the smallest side of the image during testing
_C.INPUT.HEIGHT_TEST = 384
# Maximum size of the side of the image during testing
_C.INPUT.WIDTH_TEST = 1280

# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]  # kitti
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]  # kitti
# Convert image to BGR format
_C.INPUT.TO_BGR = False
_C.INPUT.MODIFY_ALPHA = False

_C.INPUT.USE_APPROX_CENTER = False

_C.INPUT.HEATMAP_CENTER = '3D'
_C.INPUT.ADJUST_DIM_HEATMAP = False
_C.INPUT.ADJUST_BOUNDARY_HEATMAP = False
_C.INPUT.HEATMAP_RATIO = 0.5
_C.INPUT.ELLIP_GAUSSIAN = False

_C.INPUT.IGNORE_DONT_CARE = False

_C.INPUT.KEYPOINT_VISIBLE_MODIFY = False
_C.INPUT.ALLOW_OUTSIDE_CENTER = False
_C.INPUT.APPROX_3D_CENTER = 'intersect' # or clamp
_C.INPUT.ORIENTATION = 'head-axis' # multi-bin
_C.INPUT.ORIENTATION_BIN_SIZE = 4 # multi-bin


# aug parameters, in fact only random horizontal flip is applied
_C.INPUT.AUG_PARAMS = [[0.5]]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
# train split tor dataset
_C.DATASETS.TRAIN_SPLIT = ""
# test split for dataset
_C.DATASETS.TEST_SPLIT = ""
_C.DATASETS.DETECT_CLASSES = ("Car", "Pedestrian", "Cyclist")

# filter some unreasonable annotations of objects, truncation / min_size (2D box)
_C.DATASETS.FILTER_ANNO_ENABLE = False
_C.DATASETS.FILTER_ANNOS = [0.9, 20]

_C.DATASETS.USE_RIGHT_IMAGE = False
_C.DATASETS.CONSIDER_OUTSIDE_OBJS = False

_C.DATASETS.MAX_OBJECTS = 40

_C.DATASETS.MIN_RADIUS = 0.0
_C.DATASETS.MAX_RADIUS = 0.0
_C.DATASETS.CENTER_RADIUS_RATIO = 0.1


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
_C.MODEL.BACKBONE.CONV_BODY = "dla34"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 0
# Normalization for backbone
_C.MODEL.BACKBONE.DOWN_RATIO = 4

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# Heatmap Head options
# ---------------------------------------------------------------------------- #

# --------------------------SMOKE Head--------------------------------
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.PREDICTOR = "Base_Predictor"
_C.MODEL.HEAD.CENTER_AGGREGATION = False

# smooth_l1
_C.MODEL.HEAD.LOSS_TYPE = ["Penalty_Reduced_FocalLoss", "L1", "giou", "berhu"]

# centernet or fcos
_C.MODEL.HEAD.HEATMAP_TYPE = 'centernet'

# for vanilla focal loss
_C.MODEL.HEAD.LOSS_ALPHA = 0.25
_C.MODEL.HEAD.LOSS_GAMMA = 2

# for penalty-reduced focal loss
_C.MODEL.HEAD.LOSS_PENALTY_ALPHA = 2
_C.MODEL.HEAD.LOSS_BETA = 4

# 2d offset, 2d dimension
_C.MODEL.HEAD.NUM_CHANNEL = 256
_C.MODEL.HEAD.USE_NORMALIZATION = "BN"
_C.MODEL.HEAD.REGRESSION_HEADS = [['2d_dim'], ['3d_offset'], ['3d_dim'], ['ori_cls', 'ori_offset'], ['depth']]
_C.MODEL.HEAD.REGRESSION_CHANNELS = [[4, ], [2, ], [3, ], [4, 2], [1, ]]

_C.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH = False

_C.MODEL.HEAD.BIAS_BEFORE_BN = False
_C.MODEL.HEAD.BN_MOMENTUM = 0.1

_C.MODEL.HEAD.UNCERTAINTY_INIT = True
_C.MODEL.HEAD.UNCERTAINTY_RANGE = [-10, 10]
_C.MODEL.HEAD.UNCERTAINTY_WEIGHT = 1.0

_C.MODEL.HEAD.KEYPOINT_LOSS = 'L1'
_C.MODEL.HEAD.KEYPOINT_NORM_FACTOR = 1.0

_C.MODEL.HEAD.CORNER_LOSS_DEPTH = 'direct'

_C.MODEL.HEAD.KEYPOINT_XY_WEIGHT = [1, 1]
_C.MODEL.HEAD.DEPTH_FROM_KEYPOINT = False
_C.MODEL.HEAD.KEYPOINT_TO_DEPTH_RELU = True

_C.MODEL.HEAD.DEPTH_MODE = 'exp' # or linear
_C.MODEL.HEAD.DEPTH_RANGE = [0.1, 100]
_C.MODEL.HEAD.DEPTH_REFERENCE = (26.494627, 16.05988)

_C.MODEL.HEAD.SUPERVISE_CORNER_DEPTH = False
_C.MODEL.HEAD.REGRESSION_OFFSET_STAT = [-0.5844396972302358, 9.075032501413093]
_C.MODEL.HEAD.REGRESSION_OFFSET_STAT_NORMAL = [-0.01571878324572745, 0.05915441457040611]

# ['hm_loss', 'center_loss', 'bbox_loss', 'depth_loss', 'locs_loss', 'orien_loss', 'dims_loss']
_C.MODEL.HEAD.USE_UNCERTAINTY = False

_C.MODEL.HEAD.LOSS_NAMES = ['hm_loss', 'center_loss', 'bbox_loss', 'depth_loss', 'offset_loss', 'orien_loss', 'dims_loss', 'corner_loss']
_C.MODEL.HEAD.LOSS_UNCERTAINTY = [True, True, True, False, False, True, True, True]

_C.MODEL.HEAD.INIT_LOSS_WEIGHT = []
_C.MODEL.HEAD.REGRESSION_AREA = False

# edge fusion module 
_C.MODEL.HEAD.ENABLE_EDGE_FUSION = False
_C.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE = 3
_C.MODEL.HEAD.EDGE_FUSION_NORM = 'BN'
_C.MODEL.HEAD.EDGE_FUSION_RELU = False

_C.MODEL.HEAD.TRUNCATION_OFFSET_LOSS = 'L1'
_C.MODEL.HEAD.TRUNCATION_OUTPUT_FUSION = 'replace'

_C.MODEL.HEAD.TRUNCATION_CLS = False
_C.MODEL.HEAD.OUTPUT_DEPTH = 'direct'


# Reference car size in (length, height, width)
# for (car, pedestrian, cyclist)
_C.MODEL.HEAD.DIMENSION_MEAN = ((3.8840, 1.5261, 1.6286),
                               (0.8423, 1.7607, 0.6602),
                               (1.7635, 1.7372, 0.5968))

# since only car and pedestrian have enough samples and are evaluated in KITTI server 
_C.MODEL.HEAD.DIMENSION_STD = ((0.4259, 0.1367, 0.1022),
								(0.2349, 0.1133, 0.1427),
								(0.1766, 0.0948, 0.1242))

# linear or log ; use mean or not ; use std or not
_C.MODEL.HEAD.DIMENSION_REG = ['linear', True, False]
# dimension weight for h, w, l, we expect the weight of height to be larger ?
_C.MODEL.HEAD.DIMENSION_WEIGHT = [1, 1, 1]

# adjust the ensemble loss with uncertainty

_C.MODEL.DEPTH_REFINE = CN()
_C.MODEL.DEPTH_REFINE.ENABLE = False
# detach depth when refine, so that they are not entangled
_C.MODEL.DEPTH_REFINE.DETACH_DEPTH = True
_C.MODEL.DEPTH_REFINE.USE_EARLY_FEAT = True
# when the depth is not optimized enough, the refine part makes no sense
_C.MODEL.DEPTH_REFINE.REFINE_THRESH_TYPE = '2D'
_C.MODEL.DEPTH_REFINE.REFINE_THRESH = 0.2
_C.MODEL.DEPTH_REFINE.NUM_CHANNEL = [64, 128]
_C.MODEL.DEPTH_REFINE.OUTPUT_SIZE = [14, 14]
_C.MODEL.DEPTH_REFINE.JITTER = [2, 1]
_C.MODEL.DEPTH_REFINE.BIN_NUM = 5
_C.MODEL.DEPTH_REFINE.BIN_SIZE = 1

_C.MODEL.HEAD.INIT_P = 0.01

# centerness works for points 'inside' bboxes or only in the 'center' area of bboxes
_C.MODEL.HEAD.CENTER_SAMPLE = 'center'

"""
when overlaps occur, choose 
the maximum value of centerness: 'max'
or 
the one corresponding to smaller objects: 'area'
"""
_C.MODEL.HEAD.CENTER_MODE = 'max' 

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "adamw"
_C.SOLVER.BASE_LR = 3e-3
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.MAX_ITERATION = 30000 # total steps in iterations
_C.SOLVER.MAX_EPOCHS = 70 # total steps in epochs

# AdamOneCycle, not working
_C.SOLVER.MOMS = [0.95, 0.85] # larger lr <-> smaller momentum
_C.SOLVER.PCT_START = 0.4 # the percentage of rising lr, e.g. 0.4 means the lr is increasing for 40% of iterations and decreasing for the other 60%.
_C.SOLVER.DIV_FACTOR = 10 # cycle = max_lr <-> max_lr / div_factor

# For multi-step decay
_C.SOLVER.STEPS = (20000, 25000) # steps in iterations
_C.SOLVER.DECAY_EPOCH_STEPS = [35, 45] # steps in epochs
_C.SOLVER.LR_DECAY = 0.1 # lr = lr * lr_decay

_C.SOLVER.LR_CLIP = 0.0000001 # minimal learning rate

# warming up
_C.SOLVER.LR_WARMUP = False
_C.SOLVER.WARMUP_EPOCH = 1
_C.SOLVER.WARMUP_STEPS = -1

# grad clip, not used
_C.SOLVER.GRAD_NORM_CLIP = -1

_C.SOLVER.SAVE_CHECKPOINT_INTERVAL = 1000
_C.SOLVER.EVAL_INTERVAL = 2000
_C.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL = 5
_C.SOLVER.EVAL_EPOCH_INTERVAL = 2

# convert iterations to epochs for saving and evaluation
_C.SOLVER.EVAL_AND_SAVE_EPOCH = False

# deprecated
_C.SOLVER.GRAD_CLIP_FACTOR = 99
_C.SOLVER.GRAD_ALPHA = 0.9

###### LR FACTOR ######
_C.SOLVER.BIAS_LR_FACTOR = 2.0 # bias lr
_C.SOLVER.BACKBONE_LR_FACTOR = 1.0

_C.SOLVER.LOAD_OPTIMIZER_SCHEDULER = True

# Number of images per batch
_C.SOLVER.IMS_PER_BATCH = 32
_C.SOLVER.MASTER_BATCH = -1

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.SINGLE_GPU_TEST = True
_C.TEST.IMS_PER_BATCH = 1
_C.TEST.PRED_2D = True

_C.TEST.UNCERTAINTY_AS_CONFIDENCE = False

_C.TEST.METRIC = ['R40']
_C.TEST.EVAL_DIS_IOUS = False
_C.TEST.EVAL_DEPTH = False

_C.TEST.EVAL_DEPTH_METHODS = []

# 'none', '2d', '3d'
_C.TEST.USE_NMS = 'none'
_C.TEST.NMS_THRESH = -1.
_C.TEST.NMS_CLASS_AGNOSTIC = False

# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 50
_C.TEST.DETECTIONS_THRESHOLD = 0.1
_C.TEST.VISUALIZE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./tools/logs"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = -1

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True
_C.START_TIME = 0

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
