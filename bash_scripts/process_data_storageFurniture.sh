# first, use blenderproc to render the images
export CUDA_VISIBLE_DEVICES=0,1,2,3 # specify the GPU to use
export OPENCV_IO_ENABLE_OPENEXR=1 # enable EXR support in OpenCV imageio plugin if available
LOOPS=2
FRAME=6
OUT_DIR=/mnt/data/zhangzhaodong/real2code/datasets/output
MOBILITY_DATA_DIR=/mnt/data/zhangzhaodong/real2code/datasets/mobility_dataset_v2

OBJ=StorageFurniture 

SPLIT=test
export MB_DATADIR=${MOBILITY_DATA_DIR}/${SPLIT}/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}  --split $SPLIT --out_dir $OUT_DIR -o --render_bg --num_loops $LOOPS --num_frames $FRAME
done 

SPLIT=train
export MB_DATADIR=${MOBILITY_DATA_DIR}/${SPLIT}/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}  --split $SPLIT --out_dir $OUT_DIR -o --render_bg --num_loops $LOOPS --num_frames $FRAME
done 