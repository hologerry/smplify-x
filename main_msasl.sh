

python main.py --config cfg_files/fit_smplx.yaml \
    --data_folder /D_data/SL/data/MSASL/ \
    --output_folder smplx_msasl_output \
    --visualize False \
    --model_folder ../smplx_models/models \
    --vposer_ckpt ../human_body_prior/support_data/dowloads/vposer_v1_0 \
    --part_segm_fn smplx_parts_segm.pkl \
    --part_idx $1 \
    --part_num $2 \
