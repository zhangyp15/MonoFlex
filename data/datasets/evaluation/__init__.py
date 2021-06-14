import pdb
from data import datasets

from data.datasets.evaluation.kitti.kitti_eval import kitti_evaluation
from .kitti_object_eval_python.evaluate import evaluate as _evaluate_python
from .kitti_object_eval_python.evaluate import generate_kitti_3d_detection, check_last_line_break

def evaluate(eval_type, dataset, predictions, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        eval_type:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        eval_type=eval_type,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,

    )
    if isinstance(dataset, datasets.KITTIDataset):
        return kitti_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))

def evaluate_python(label_path, result_path, label_split_file, current_class, metric):
    result, ret_dict = _evaluate_python(label_path, result_path, label_split_file, current_class, metric=metric)

    return result, ret_dict

if __name__ == '__main__':
    pass