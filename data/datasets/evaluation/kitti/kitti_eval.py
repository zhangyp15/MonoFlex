import os
import csv
import logging
import subprocess
import pdb
import shutil

from utils.miscellaneous import mkdir

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}

def kitti_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger("monoflex.kitti_eval")
    if "detection" in eval_type:
        logger.info("performing kitti detection evaluation: ")
        result_dict = do_kitti_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )

        return result_dict


def do_kitti_detection_evaluation(dataset,
                                  predictions,
                                  output_folder,
                                  logger
                                  ):
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    # remove generated files from previous evaluation
    for file in os.listdir(output_folder):
        if file == 'data':
            continue
        file = os.path.join(output_folder, file)
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)

    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(predict_folder, predict_txt)

        generate_kitti_3d_detection(prediction, predict_txt)

    logger.info("Evaluate on KITTI dataset")
    cwd = os.getcwd()
    output_dir = os.path.abspath(output_folder)
    os.chdir('./data/datasets/evaluation/kitti')
    label_dir = os.path.join(cwd, getattr(dataset, 'label_dir'))
    if not os.path.isfile('evaluate_object_3d_offline'):
        subprocess.Popen('g++ -O3 -DNDEBUG -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp', shell=True)

    command = "./evaluate_object_3d_offline {} {}".format(label_dir, output_dir)
    print('evaluating with command {}'.format(command))
    output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()

    # filter output info
    output = output.splitlines(keepends=False)
    # remove save .... info
    result_dict = {}
    match_str = 'AP: '
    for info in output:
        if info.find(match_str) >= 0:
            logger.info(info)
            info_split = info.split(match_str)
            key = info_split[0].rstrip()
            value = info_split[1].split()
            result_dict[key] = value
        
    os.chdir(os.path.join(cwd))

    return result_dict

def generate_kitti_3d_detection(prediction, predict_txt):
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
