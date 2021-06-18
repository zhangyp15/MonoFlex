import argparse
import os
import json

from tools.evaluation.kitti_utils.eval import kitti_eval
from tools.evaluation.kitti_utils import kitti_common as kitti

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Evaluate mAP in KITTI format.')
parser.add_argument('--pred_label_folder', type=str, default="./", help='the path of pred label folder in kitti_format')
args = parser.parse_args()

if __name__ == '__main__':
    pred_label_folder = args.pred_label_folder
    gt_label_path = "/root/SMOKE/datasets/kitti/training/label_2"
    metric_path = os.path.join(pred_label_folder, "../R11")
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    
    val_mAP = []
    for pred_path in os.listdir(pred_label_folder):
        print("pred_path: ", pred_path)
        iteration = int(pred_path.split('_')[1])
        pred_label_path = os.path.join(pred_label_folder, pred_path, "data")
        print(pred_label_path)
        
        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"])
        
        if ret_dict is not None:
            mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
            val_mAP.append(mAP_3d_moderate)
            
            with open(os.path.join(metric_path, "val_mAP.json"),'w') as file_object:
                json.dump(val_mAP, file_object)
            with open(os.path.join(metric_path, 'epoch_result_{:07d}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
                f.write(result)
            print(result)
