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
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
# Contact: Vassilis choutas, vassilis.choutas@tuebingen.mpg.de

import argparse
import os
import os.path as osp
import pickle

import cv2
import numpy as np
import PIL.Image as pil_img
import pyrender
import smplx
import torch
import trimesh

from human_body_prior.tools.model_loader import load_vposer
from numpy import pi
from pyrender.light import DirectionalLight
from pyrender.node import Node
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm

from cmd_parser import parse_config
from fit_single_frame import imshow_keypoints
from utils import JointMapper


def _create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(Node(light=DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

    return nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args, remaining = parser.parse_known_args()
    args = parse_config(remaining)

    dtype = torch.float32
    use_cuda = args.get("use_cuda", True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_type = args.get("model_type", "smplx")
    print(f"Model type: {model_type}, folder: {args.get('model_folder')}")

    model_params = dict(
        model_path=args.get("model_folder"),
        #  joint_mapper=joint_mapper,
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

    model = smplx.create(**model_params)
    model = model.to(device=device)

    batch_size = args.get("batch_size", 1)
    use_vposer = args.get("use_vposer", True)
    vposer, pose_embedding = None, None
    vposer_ckpt = args.get("vposer_ckpt", "")
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32], dtype=dtype, device=device, requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model="snapshot")
        vposer = vposer.to(device=device)
        vposer.eval()

    data_folder = args.get("data_folder")
    output_folder = args.get("output_folder")

    print(f"Data folder: {data_folder}, output folder: {output_folder}")

    coco_wholebody_pkl = os.path.join(data_folder, "keypoints_hrnet_dark_coco_wholebody.pkl")
    with open(coco_wholebody_pkl, "rb") as f:
        keypoints_all = pickle.load(f)

    camera_default_transl = [0.0, -0.7, 24.0]
    W, H = 256, 256

    os.makedirs(f"{output_folder}/img_meanpose", exist_ok=True)

    frames_sub_name = "frames1" if "MSASL" in data_folder else "frames"

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

    for vid_dict in tqdm(cur_split_dicts, total=len(cur_split_dicts), leave=False):
        # print("video", video_id)
        video_name = vid_dict["name"]
        seq_len = vid_dict["seq_len"]
        pkl_path = os.path.join(output_folder, "results", video_name)
        pkl_files = sorted(os.listdir(pkl_path))
        num_pkl_files = len(pkl_files)

        cur_vid_mean_pose_img_folder = os.path.join(output_folder, "img_meanpose", video_name)

        os.makedirs(cur_vid_mean_pose_img_folder, exist_ok=True)

        est_params = {}
        for pkl_name in pkl_files:
            with open(os.path.join(pkl_path, pkl_name), "rb") as f:
                data = pickle.load(f)
            for key, val in data[0]["result"].items():
                if key not in est_params:
                    est_params[key] = []
                est_params[key].append(val)

        for key, val in est_params.items():
            cat_val = np.concatenate(val, axis=0)
            # smooth across frames
            est_params[key] = gaussian_filter1d(cat_val, sigma=1, axis=0)

        for idx, pkl_name in tqdm(enumerate(pkl_files), total=len(pkl_files), desc=video_name):

            with open(os.path.join(pkl_path, pkl_name), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if use_vposer:
                with torch.no_grad():
                    pose_embedding[:] = torch.tensor(
                        est_params["body_pose"][idx : idx + 1], device=device, dtype=dtype
                    )

            cur_frame_est_params = {}
            for key, val in data[0]["result"].items():

                if key == "body_pose" and use_vposer:
                    body_pose = vposer.decode(pose_embedding, output_type="aa").view(1, -1)
                    # if model_type == "smpl":
                    #     print("model_type is smpl, add wrist pose")
                    #     wrist_pose = torch.zeros([body_pose.shape[0], 6], dtype=dtype, device=device)
                    #     body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                    cur_frame_est_params["body_pose"] = body_pose
                elif key == "betas":
                    # use the mean shape (betas)
                    cur_frame_est_params[key] = torch.zeros([1, 10], dtype=dtype, device=device)
                else:
                    cur_frame_est_params[key] = torch.tensor(
                        est_params[key][idx : idx + 1], dtype=dtype, device=device
                    )

            model_output = model(**cur_frame_est_params)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()

            out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
            )
            mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, "mesh")

            light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
            # light_pose = np.eye(4)
            light_pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            scene.add(light, pose=light_pose)

            light_nodes = _create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)

            camera_center = [128.0, 128.0]

            focal_length = 5000
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length, cx=camera_center[0], cy=camera_center[1]
            )

            front_camera_transl = np.zeros(3)
            front_camera_transl[0] = camera_default_transl[0]
            front_camera_transl[1] = camera_default_transl[1]
            front_camera_transl[2] = -1.0 * camera_default_transl[2]
            front_camera_pose = np.eye(4)
            front_camera_pose[:3, 3] = front_camera_transl
            front_camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

            camera_node = scene.add(camera, pose=front_camera_pose)

            r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
            front_color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

            front_color = pil_img.fromarray(front_color)
            front_img_path = os.path.join(cur_vid_mean_pose_img_folder, f"{idx:03d}_front.png")
            front_color.save(front_img_path)

            side_camera_transl = np.zeros(3)
            side_camera_transl[0] = camera_default_transl[2]
            side_camera_transl[1] = camera_default_transl[1]
            side_camera_transl[2] = -1.0 * camera_default_transl[0]

            side_camera_pose = np.eye(4)
            side_camera_pose[:3, 3] = side_camera_transl
            side_camera_pose[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]

            scene.set_pose(camera_node, pose=side_camera_pose)

            side_color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

            side_color = pil_img.fromarray(side_color)
            side_img_path = os.path.join(cur_vid_mean_pose_img_folder, f"{idx:03d}_side.png")
            side_color.save(side_img_path)

        keypoints = np.array(keypoints_all[video_name])

        cap_fps = 13

        images_path = os.path.join(data_folder, frames_sub_name, video_name)
        images_names = sorted(os.listdir(images_path))
        n_frames = len(images_names)

        res_video_path = os.path.join(f"{output_folder}", "results_meanpose")
        os.makedirs(res_video_path, exist_ok=True)
        out_video_path = os.path.join(res_video_path, f"{video_name}_meanpose.mp4")

        tmp_folder = os.path.join("/tmp/", video_name)
        os.makedirs(tmp_folder, exist_ok=True)
        for frame_idx in range(1, n_frames - 1):
            if keypoints[frame_idx - 1, 10, 2] < 0.7 and keypoints[frame_idx - 1, 9, 2] < 0.7:
                continue

            file0 = os.path.join(images_path, images_names[frame_idx])
            file1 = os.path.join(cur_vid_mean_pose_img_folder, f"{frame_idx:03d}_front.png")
            file2 = os.path.join(cur_vid_mean_pose_img_folder, f"{frame_idx:03d}_side.png")

            img0 = cv2.imread(file0)
            img0_kp = imshow_keypoints(img0, keypoints[frame_idx - 1], kpt_score_thr=0.5)
            img1 = cv2.imread(file1)
            img2 = cv2.imread(file2)

            out_frame = np.zeros((256, 768, 3), dtype=np.uint8)
            out_frame[:256, :256, :] = img0_kp[:, :, :]
            out_frame[:256, 256:512, :] = img1[:, :, :]
            out_frame[:256, 512:, :] = img2[:, :, :]

            cv2.imwrite(os.path.join(tmp_folder, f"{frame_idx:03d}.png"), out_frame)

        cmd = f"ffmpeg -hide_banner -loglevel error -framerate {cap_fps} -pattern_type glob -i '{tmp_folder}/*.png' -c:v libx264 -pix_fmt yuv420p -y {out_video_path}"
        os.system(cmd)
