import math
import random
import pdb
import copy
import numpy as np
import torch
from PIL import Image, ImageOps
from data.datasets.kitti_utils import convertRot2Alpha, convertAlpha2Rot, refresh_attributes
from transforms.box import box_iou
from model.heatmap_coder import get_transfrom_matrix,affine_transform


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

        return img, objs, calib

class RandomAffineCrop(object):
    def __init__(self,p):
        self.p = p
        self.input_width, self.input_height = 1280, 384
        self.shift_scale = [0.2,0.2]

    def __call__(self, img, objs, calib):
        #img.save("/home/lipengcheng/results/kitti_test/ori.png")
        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        shift, scale = self.shift_scale[0], self.shift_scale[1]
        shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
        #center[0] += size[0] * random.choice(shift_ranges)
        center[1] += size[1] * random.choice(shift_ranges)

        scale_ranges = np.arange(1 - scale, 1 , 0.1)
        size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )
        #img.save("/home/lipengcheng/results/kitti_test/affine.png")
        calib.matAndUpdate(trans_affine)
        save_id = []
        roi = torch.Tensor([0,0,self.input_width,self.input_height]).view(-1,4)
        for idx, obj in enumerate(objs):
            box2d = obj.box2d
            box2d[:2] = affine_transform(box2d[:2], trans_affine)
            box2d[2:] = affine_transform(box2d[2:], trans_affine)
            
            ious = box_iou(torch.Tensor(box2d).view(-1,4), roi)
            if ious > 0 :
                save_id.append(idx)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.input_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.input_height - 1)
            obj.box2d = box2d
            
            objs[idx] = obj
        
        objs = [objs[id] for id in save_id]


        # if random.random() < self.p:

        #     img_w, img_h = img.size

        #     # flip labels
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

        #         # projection-based 3D center flip
        #         # center_loc = obj.t.copy()
        #         # center_loc[1] -= obj.h / 2
        #         # center2d, depth = calib.project_rect_to_image(center_loc.reshape(1, 3))
        #         # center2d[:, 0] = img_w - center2d[:, 0] - 1
        #         # center3d = flip_calib.project_image_to_rect(np.concatenate([center2d, depth.reshape(-1, 1)], axis=1))[0]
        #         # center3d[1] += obj.h / 2
                
        #         # fliped 3D center
        #         loc = obj.t.copy()
        #         loc[0] = -loc[0]
        #         obj.t = loc
                
        #         obj.alpha = convertRot2Alpha(roty, obj.t[2], obj.t[0])
        #         objs[idx] = obj

        #     # flip calib
        #     P2 = calib.P.copy()
        #     P2[0, 2] = img_w - P2[0, 2] - 1
        #     P2[0, 3] = - P2[0, 3]
        #     calib.P = P2
        #     refresh_attributes(calib)

        return img, objs, calib