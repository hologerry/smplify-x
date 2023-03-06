# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import smplx
import torch
import yaml

from tqdm import tqdm

from camera import create_camera
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame
from prior import create_prior
from utils import JointMapper


torch.backends.cudnn.enabled = False


def convert_coco_keypoints_to_openpose(coco_kps):
    op_keypoints = np.zeros([coco_kps.shape[0], 118, 3])
    op_keypoints[:, 0, :] = coco_kps[:, 0, :]
    op_keypoints[:, 1, 1:] = coco_kps[:, 6, 1:]
    op_keypoints[:, 1, 0] = (coco_kps[:, 5, 0] + coco_kps[:, 6, 0]) / 2
    op_keypoints[:, 2, :] = coco_kps[:, 6, :]
    op_keypoints[:, 3, :] = coco_kps[:, 8, :]
    op_keypoints[:, 4, :] = coco_kps[:, 10, :]
    op_keypoints[:, 5, :] = coco_kps[:, 5, :]
    op_keypoints[:, 6, :] = coco_kps[:, 7, :]
    op_keypoints[:, 7, :] = coco_kps[:, 9, :]
    op_keypoints[:, 8, 1:] = coco_kps[:, 12, 1:]
    op_keypoints[:, 8, 0] = (coco_kps[:, 12, 0] + coco_kps[:, 11, 0]) / 2
    op_keypoints[:, 9, :] = coco_kps[:, 12, :]
    op_keypoints[:, 10, :] = coco_kps[:, 14, :]
    op_keypoints[:, 11, :] = coco_kps[:, 16, :]
    op_keypoints[:, 12, :] = coco_kps[:, 11, :]
    op_keypoints[:, 13, :] = coco_kps[:, 13, :]
    op_keypoints[:, 14, :] = coco_kps[:, 15, :]
    op_keypoints[:, 15, :] = coco_kps[:, 2, :]
    op_keypoints[:, 16, :] = coco_kps[:, 1, :]
    op_keypoints[:, 17, :] = coco_kps[:, 4, :]
    op_keypoints[:, 18, :] = coco_kps[:, 3, :]
    op_keypoints[:, 19, :] = coco_kps[:, 17, :]
    op_keypoints[:, 20, :] = coco_kps[:, 18, :]
    op_keypoints[:, 21, :] = coco_kps[:, 19, :]
    op_keypoints[:, 22, :] = coco_kps[:, 21, :]
    op_keypoints[:, 23, :] = coco_kps[:, 21, :]
    op_keypoints[:, 24, :] = coco_kps[:, 22, :]
    op_keypoints[:, 25:67, :] = coco_kps[:, 91:, :]
    op_keypoints[:, 67:, :] = coco_kps[:, 40:91, :]
    for i_ in range(op_keypoints.shape[0]):
        for j_ in range(118):
            if op_keypoints[i_, j_, 2] < 0.65:
                op_keypoints[i_, j_, 2] = 0.0

    return op_keypoints


def main(**args):

    args.pop("img_folder")
    data_folder = args.pop("data_folder")
    output_folder = args.pop("output_folder")
    output_folder = osp.expandvars(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, "conf.yaml")
    with open(conf_fn, "w") as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop("result_folder", "results")
    result_folder = osp.join(output_folder, result_folder)
    os.makedirs(result_folder, exist_ok=True)

    mesh_folder = args.pop("mesh_folder", "meshes")
    mesh_folder = osp.join(output_folder, mesh_folder)
    os.makedirs(mesh_folder, exist_ok=True)

    out_img_folder = osp.join(output_folder, "images")
    os.makedirs(out_img_folder, exist_ok=True)

    float_dtype = args["float_dtype"]
    if float_dtype == "float64":
        dtype = torch.float64
    elif float_dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown float type {float_dtype}, exiting!")

    use_cuda = args.get("use_cuda", True)
    if use_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, exiting!")

    coco_wholebody_pkl = os.path.join(data_folder, "keypoints_hrnet_dark_coco_wholebody.pkl")
    with open(coco_wholebody_pkl, "rb") as f:
        coco_wholebody_dict = pickle.load(f)

    split = args.get("split", "test")
    split_file = os.path.join(data_folder, f"{split}.pkl")
    with open(split_file, "rb") as f:
        split_dicts = pickle.load(f)

    num_process_videos = args.get("num_process_videos", -1)
    if num_process_videos > 0:
        cur_split_dicts = split_dicts[:num_process_videos]
    else:
        part_idx = args.get("part_idx", 0)
        part_num = args.get("part_num", 1)
        cur_split_dicts = split_dicts[part_idx::part_num]

    nan_flag = False

    for vid_dict in tqdm(cur_split_dicts, total=len(cur_split_dicts), leave=False, desc="Processing videos"):
        video_name = vid_dict["name"]
        seq_len = vid_dict["seq_len"]

        frames_sub_name = "frames"

        frames_folder = os.path.join(data_folder, frames_sub_name, video_name)
        dataset_obj = create_dataset(img_folder=frames_folder, **args)
        assert len(dataset_obj) == seq_len
        start = time.time()

        input_gender = args.pop("gender", "neutral")
        # gender_lbl_type = args.pop("gender_lbl_type", "none")

        joint_mapper = JointMapper(dataset_obj.get_model2data())

        model_params = dict(
            model_path=args.get("model_folder"),
            joint_mapper=joint_mapper,
            create_global_orient=True,
            create_body_pose=not args.get("use_vposer"),
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=False,
            dtype=dtype,
            **args,
        )

        male_model = smplx.create(gender="male", **model_params)
        # SMPL-H has no gender-neutral model
        if args.get("model_type") != "smplh":
            neutral_model = smplx.create(gender="neutral", **model_params)
        female_model = smplx.create(gender="female", **model_params)

        # Create the camera object
        focal_length = args.get("focal_length")
        camera = create_camera(focal_length_x=focal_length, focal_length_y=focal_length, dtype=dtype, **args)

        if hasattr(camera, "rotation"):
            camera.rotation.requires_grad = False

        use_hands = args.get("use_hands", True)
        use_face = args.get("use_face", True)

        body_pose_prior = create_prior(prior_type=args.get("body_prior_type"), dtype=dtype, **args)

        jaw_prior, expr_prior = None, None
        if use_face:
            jaw_prior = create_prior(prior_type=args.get("jaw_prior_type"), dtype=dtype, **args)
            expr_prior = create_prior(prior_type=args.get("expr_prior_type", "l2"), dtype=dtype, **args)

        left_hand_prior, right_hand_prior = None, None
        if use_hands:
            lhand_args = args.copy()
            lhand_args["num_gaussians"] = args.get("num_pca_comps")
            left_hand_prior = create_prior(
                prior_type=args.get("left_hand_prior_type"), dtype=dtype, use_left_hand=True, **lhand_args
            )

            rhand_args = args.copy()
            rhand_args["num_gaussians"] = args.get("num_pca_comps")
            right_hand_prior = create_prior(
                prior_type=args.get("right_hand_prior_type"), dtype=dtype, use_right_hand=True, **rhand_args
            )

        shape_prior = create_prior(prior_type=args.get("shape_prior_type", "l2"), dtype=dtype, **args)

        angle_prior = create_prior(prior_type="angle", dtype=dtype)


        # joints_fix = torch.zeros([9, 3])
        camera_transl = torch.zeros([3])
        camera_orient = torch.zeros([3])
        betas_fix = torch.zeros([10])
        prev_body_pose = torch.zeros([1, 21, 3])

        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")

            # joints_fix = joints_fix.to(device=device)
            camera_transl = camera_transl.to(device=device)
            camera_orient = camera_orient.to(device=device)
            betas_fix = betas_fix.to(device=device)
            prev_body_pose = prev_body_pose.to(device=device)

            camera = camera.to(device=device)
            female_model = female_model.to(device=device)
            male_model = male_model.to(device=device)
            if args.get("model_type") != "smplh":
                neutral_model = neutral_model.to(device=device)
            body_pose_prior = body_pose_prior.to(device=device)
            angle_prior = angle_prior.to(device=device)
            shape_prior = shape_prior.to(device=device)
            if use_face:
                expr_prior = expr_prior.to(device=device)
                jaw_prior = jaw_prior.to(device=device)
            if use_hands:
                left_hand_prior = left_hand_prior.to(device=device)
                right_hand_prior = right_hand_prior.to(device=device)
        else:
            device = torch.device("cpu")

        gender = input_gender

        if gender == "neutral":
            body_model = neutral_model
        elif gender == "female":
            body_model = female_model
        elif gender == "male":
            body_model = male_model

        print("body_model", body_model)

        # A weight for every joint of the model
        joint_weights = dataset_obj.get_joint_weights().to(device=device, dtype=dtype)
        # Add a fake batch dimension for broadcasting
        joint_weights.unsqueeze_(dim=0)

        coco_kps = np.array(coco_wholebody_dict[video_name])

        op_keypoints = convert_coco_keypoints_to_openpose(coco_kps)

        cur_video_result_folder = osp.join(result_folder, video_name)
        os.makedirs(cur_video_result_folder, exist_ok=True)
        cur_video_mesh_folder = osp.join(mesh_folder, video_name)
        os.makedirs(cur_video_mesh_folder, exist_ok=True)

        nan_flag = False
        for idx, data in enumerate(dataset_obj):

            img = data["img"]
            fn = data["fn"]

            if idx >= op_keypoints.shape[0]:
                break
            keypoints = op_keypoints[idx : idx + 1]

            print(f"Processing: {data['img_path']}")

            curr_result_fn = osp.join(cur_video_result_folder, f"{idx + 1:03d}.pkl")
            curr_mesh_fn = osp.join(cur_video_mesh_folder, f"{idx + 1:03d}.obj")

            (
                cur_body_pose,
                cur_camera_transl,
                cur_camera_orient,
                cur_betas_fix,
                nan_flag,
            ) = fit_single_frame(
                img,
                idx,
                keypoints,
                body_model=body_model,
                camera=camera,
                joint_weights=joint_weights,
                dtype=dtype,
                result_fn=curr_result_fn,
                mesh_fn=curr_mesh_fn,
                shape_prior=shape_prior,
                expr_prior=expr_prior,
                body_pose_prior=body_pose_prior,
                left_hand_prior=left_hand_prior,
                right_hand_prior=right_hand_prior,
                jaw_prior=jaw_prior,
                angle_prior=angle_prior,
                prev_body_pose=prev_body_pose,
                camera_transl=camera_transl,
                camera_orient=camera_orient,
                betas_fix=betas_fix,
                **args,
            )
            prev_body_pose = cur_body_pose.clone()
            camera_transl = cur_camera_transl.clone()
            camera_orient = cur_camera_orient.clone()
            betas_fix = cur_betas_fix.clone()

            if nan_flag:
                break

        elapsed = time.time() - start
        time_msg = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(elapsed))
        print("Processing the data took: {}".format(time_msg))
        if nan_flag:
            continue

        with open(os.path.join(cur_video_mesh_folder, "camera.pkl"), "wb") as f:
            pickle.dump(camera_transl.cpu(), f)


if __name__ == "__main__":
    args = parse_config()
    # print(args)
    main(**args)
