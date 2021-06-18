CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --batch_size 8 --backbone RESNET-18 --output output/RESNET18_Baseline_005nd
python tools/evaluate_script.py --pred_label_folder ../output/RESNET18_Baseline_005nd/kitti_train

CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --batch_size 8 --backbone RESNET-18 --output output/RESNET18_Baseline_006nd
python tools/evaluate_script.py --pred_label_folder ../output/RESNET18_Baseline_006nd/kitti_train

CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --batch_size 8 --backbone RESNET-18 --output output/RESNET18_Baseline_007nd
python tools/evaluate_script.py --pred_label_folder ../output/RESNET18_Baseline_007nd/kitti_train

CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --batch_size 8 --backbone RESNET-18 --output output/RESNET18_Baseline_008nd
python tools/evaluate_script.py --pred_label_folder ../output/RESNET18_Baseline_008nd/kitti_train