import time
import pdb
import fire

from . import kitti_common as kitti

import csv

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1,
             metric='R40'):
    
    from .eval import get_coco_eval_result, get_official_eval_result
    
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class, metric=metric)

def generate_kitti_3d_detection(prediction, predict_txt):

    ID_TYPE_CONVERSION = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
    }

    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)

def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()

if __name__ == '__main__':
    fire.Fire()