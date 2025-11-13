RUN=storageFurniture
LS=3600
python shape_complete/eval.py -r $RUN -ls $LS --save_mesh --obj_folder \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/shape_output/ \
    -es --sam_result_dir /mnt/data/zhangzhaodong/real2code/output/shape_eval_results/$RUN