python shape_complete/train.py \
    --wandb \
    --use_dp \
    -dp 8 \
    -rn storageFurniture_only \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/shape_output/ \
    --load_voxelgrid \
    -o \
    -e 200 -b 96 -vi 100 -v 50 -vis 1200 -save 600 -log 75
