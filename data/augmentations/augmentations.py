import math
import random
import pdb
import copy
import numpy as np

from PIL import Image, ImageOps
from data.datasets.kitti_utils import convertRot2Alpha, convertAlpha2Rot, refresh_attributes

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, objs, calib):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            self.PIL2Numpy = True

        for a in self.augmentations:
            img, objs, calib = a(img, objs, calib)

        if self.PIL2Numpy:
            img = np.array(img)

        return img, objs, calib

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, objs, calib):
        if random.random() < self.p:
            # flip image
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_w, img_h = img.size

            # flip calib
            # P2 = calib.P.copy()
            # P2[0, 2] = img_w - P2[0, 2] - 1
            # P2[0, 3] = - P2[0, 3]
            # flip_calib = copy.deepcopy(calib)
            # flip_calib.P = P2
            # refresh_attributes(flip_calib)

            # flip labels
            for idx, obj in enumerate(objs):
                
                # flip box2d
                w = obj.xmax - obj.xmin
                obj.xmin = img_w - obj.xmax - 1
                obj.xmax = obj.xmin + w
                obj.box2d = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)
                
                # flip roty
                roty = obj.ry
                roty = (-math.pi - roty) if roty < 0 else (math.pi - roty)
                while roty > math.pi: roty -= math.pi * 2
                while roty < (-math.pi): roty += math.pi * 2
                obj.ry = roty

                # projection-based 3D center flip
                # center_loc = obj.t.copy()
                # center_loc[1] -= obj.h / 2
                # center2d, depth = calib.project_rect_to_image(center_loc.reshape(1, 3))
                # center2d[:, 0] = img_w - center2d[:, 0] - 1
                # center3d = flip_calib.project_image_to_rect(np.concatenate([center2d, depth.reshape(-1, 1)], axis=1))[0]
                # center3d[1] += obj.h / 2
                
                # fliped 3D center
                loc = obj.t.copy()
                loc[0] = -loc[0]
                obj.t = loc
                
                obj.alpha = convertRot2Alpha(roty, obj.t[2], obj.t[0])
                objs[idx] = obj

            # flip calib
            P2 = calib.P.copy()
            P2[0, 2] = img_w - P2[0, 2] - 1
            P2[0, 3] = - P2[0, 3]
            calib.P = P2
            refresh_attributes(calib)

        #     flip_calib = copy.deepcopy(calib)
        #     if self.flip_calibration:
        #         flip_calib.P[0, 2] = img_w - calib.P[0, 2]
        #         flip_calib.P[0, 3] = img_w * calib.P[2, 3] - calib.P[0, 3]
        #         refresh_attributes(flip_calib)

        #     remove_idx = []
        #     for idx, obj in enumerate(objs):
        #         # flip box2d
        #         w = obj.xmax - obj.xmin
        #         obj.xmin = img_w - obj.xmax - 1
        #         obj.xmax = obj.xmin + w
        #         obj.box2d = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)
                
        #         # flip roty
        #         roty = obj.ry
        #         roty = (-math.pi - roty) if roty < 0 else (math.pi - roty)
        #         while roty > math.pi: roty -= math.pi * 2
        #         while roty < (-math.pi): roty += math.pi * 2
        #         obj.ry = roty
                
        #         # fliped 3D center
        #         loc = obj.t.copy()
        #         loc[1] = loc[1] - obj.h / 2
                
        #         # project the center onto image plane
        #         center2d, depth = calib.project_rect_to_image(loc.reshape(1, 3))
                
        #         center2d[:, 0] = img_w - center2d[:, 0] - 1
        #         center3d = flip_calib.project_image_to_rect(np.concatenate([center2d, depth.reshape(-1, 1)], axis=1))[0]
        #         center3d[1] += obj.h / 2
                
        #         obj.t = center3d
        #         obj.alpha = convertRot2Alpha(roty, center3d[2], center3d[0], modify_alpha=self.modify_alpha)
        #         objs[idx] = obj

        #     objs = [obj for idx, obj in enumerate(objs) if idx not in remove_idx]
        # else:
        #     flip_calib = calib

        return img, objs, calib