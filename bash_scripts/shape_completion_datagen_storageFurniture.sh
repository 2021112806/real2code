# export CUDA_VISIBLE_DEVICES=4
# python shape_complete/datagen.py --data_dir /mnt/data/zhangzhaodong/real2code/datasets/output/ --obj_type Scissors --out_dir /mnt/data/zhangzhaodong/real2code/shape_output/ --vis_dir /mnt/data/zhangzhaodong/real2code/shape_output/vis --split train

export CUDA_VISIBLE_DEVICES=4,5,6,7
OBJ=StorageFurniture
for SPLIT in train test; do
    python shape_complete/datagen.py --data_dir /mnt/data/zhangzhaodong/real2code/datasets/output/ --obj_type $OBJ --out_dir /mnt/data/zhangzhaodong/real2code/datasets/shape_output/ --vis_dir /mnt/data/zhangzhaodong/real2code/datasets/shape_output/vis --split $SPLIT
done