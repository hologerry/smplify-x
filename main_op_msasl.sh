python main_op.py --config cfg_files/fit_smplx.yaml \
    --data_folder /D_data/SL/data/MSASL/ \
    --output_folder smplx_msasl_output_op \
    --visualize False \
    --model_folder "../smplx_support_files/smplx_models/models" \
    --vposer_ckpt "../smplx_support_files/vposer_models/vposer_v1_0" \
    --part_segm_fn "../smplx_support_files/smplx_parts_segm.pkl" \
    --num_process_videos 1 \
