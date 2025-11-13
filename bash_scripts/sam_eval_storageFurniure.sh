export CUDA_VISIBLE_DEVICES=0,1,2,3
# 在test目录
python part_segmentation/eval_sam.py --obj_type StorageFurniture --obj_folder 48051 --sam_load_epoch 48 --num_3d_points 100 -o