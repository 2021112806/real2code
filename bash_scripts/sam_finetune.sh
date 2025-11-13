python part_segmentation/finetune_sam.py --blender --run_name rebuttal_full \
 --wandb --use_dp -dp 8 --batch_size 24 --epochs 100 --data_dir /mnt/data/zhangzhaodong/real2code/datasets/output

python part_segmentation/finetune_sam.py --blender --run_name scissors_eyeglasses_only \
 --wandb --use_dp -dp 8 --batch_size 24 --epochs 100 --data_dir /mnt/data/zhangzhaodong/real2code/datasets/output --rebuttal_objects