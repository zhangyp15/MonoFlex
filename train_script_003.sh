CUDA_VISIBLE_DEVICES=7 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --output output/DLA34_Baseline_001nd
python tools/evaluate_script.py --pred_label_folder ../output/DLA34_Baseline_001nd/kitti_train

CUDA_VISIBLE_DEVICES=7 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --output output/DLA34_Baseline_002nd
python tools/evaluate_script.py --pred_label_folder ../output/DLA34_Baseline_002nd/kitti_train

CUDA_VISIBLE_DEVICES=7 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --output output/DLA34_Baseline_003nd
python tools/evaluate_script.py --pred_label_folder ../output/DLA34_Baseline_003nd/kitti_train

CUDA_VISIBLE_DEVICES=7 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --output output/DLA34_Baseline_004nd
python tools/evaluate_script.py --pred_label_folder ../output/DLA34_Baseline_004nd/kitti_train