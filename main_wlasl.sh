python main.py --config cfg_files/fit_smplx.yaml \
    --data_folder /D_data/SL/data/WLASL/ \
    --output_folder smplx_wlasl_output \
    --visualize False \
    --model_folder ../smplx_models/models \
    --vposer_ckpt ../human_body_prior/support_data/downloads/vposer_v1_0 \
    --part_segm_fn smplx_parts_segm.pkl \
