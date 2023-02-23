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
from tqdm import tqdm

from cmd_parser import parse_config
from utils import JointMapper


def read_pickle(work_path):
    data_list = []
    with open(work_path, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                data_list.append(data)
            except EOFError:
                break
    return data_list


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


class GuassianBlur:
    def __init__(self, r, sigma=1):
        # 按照guassian方式构建kernel，输入参数高斯模糊半径r
        self.r = r
        self.kernel = np.empty(2 * r + 1)
        total = 0
        for i in range(2 * r + 1):
            self.kernel[i] = np.exp(-((i - r) ** 2) / (2 * sigma**2)) / ((2 * pi) ** 1 / 2 * sigma**2)
            # self.kernel[i] = 1.
            total += self.kernel[i]
        self.kernel /= total

    def guassian_blur(self, mesh, flag=0):
        b, l, k = mesh.shape
        mesh_copy = np.zeros([b + 2 * self.r, l, k])
        mesh_copy[: self.r, :, :] = mesh[0, :, :]
        mesh_copy[self.r : b + self.r, :, :] = mesh
        mesh_copy[b + self.r : b + 2 * self.r, :, :] = mesh[-1, :, :]

        for i in range(k):
            for j in range(self.r, self.r + b):
                # for m in range(k):
                mesh_copy[j, 0, i] = np.sum(self.kernel * mesh_copy[j - self.r : j + self.r + 1, 0, i])  # 卷积运算

        return mesh_copy[self.r : self.r + b, :, :]


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

    video_ids = ["00583"]

    for video_id in tqdm(video_ids, leave=False):
        # print("video", video_id)
        pkl_paths = sorted(os.listdir(f"{output_folder}/results/" + video_id))
        os.makedirs(f"{output_folder}/img_meanpose/" + video_id, exist_ok=True)
        with open(f"{output_folder}/results/" + video_id + "/" + pkl_paths[0], "rb") as f:
            data = pickle.load(f, encoding="latin1")  # first frame data
        # print(data[0].keys())
        # print(data[0]["result"].keys())

        est_params = {}
        for key, val in data[0]["result"].items():
            if key == "camera_rotation":
                data_key = np.zeros([len(pkl_paths), 1, 3, 3])
                for idx, pkl_path in enumerate(pkl_paths):
                    pkl_path = f"{output_folder}/results/" + video_id + "/" + pkl_path
                    with open(pkl_path, "rb") as f:
                        data_i = pickle.load(f, encoding="latin1")
                    data_key[idx] = data_i[0]["result"][key]
                est_params[key] = data_key
            else:
                data_key = np.zeros([len(pkl_paths), 1, data[0]["result"][key].shape[1]])
                for idx, pkl_path in enumerate(pkl_paths):
                    pkl_path = f"{output_folder}/results/" + video_id + "/" + pkl_path
                    with open(pkl_path, "rb") as f:
                        data_i = pickle.load(f, encoding="latin1")
                    data_key[idx] = data_i[0]["result"][key]
                est_params[key] = data_key

        #####blur#######

        for key, val in data[0]["result"].items():
            if key == "camera_rotation":
                data_temp = est_params[key].reshape(-1, 1, 9)
                GuassianBlur_ = GuassianBlur(2)
                out_smooth = GuassianBlur_.guassian_blur(data_temp, flag=0)
                est_params[key] = out_smooth.reshape(-1, 1, 3, 3)
            else:
                GuassianBlur_ = GuassianBlur(2)
                out_smooth = GuassianBlur_.guassian_blur(est_params[key], flag=0)
                est_params[key] = out_smooth

        for idx, pkl_path in tqdm(enumerate(pkl_paths), total=len(pkl_paths), desc=video_id):

            frame_idx = pkl_path.split(".")[0]
            pkl_path = f"{output_folder}/results/" + video_id + "/" + pkl_path
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if use_vposer:
                with torch.no_grad():
                    pose_embedding[:] = torch.tensor(est_params["body_pose"][idx], device=device, dtype=dtype)

            est_params_i = {}
            for key, val in data[0]["result"].items():

                if key == "body_pose" and use_vposer:
                    body_pose = vposer.decode(pose_embedding, output_type="aa").view(1, -1)
                    if model_type == "smpl":
                        wrist_pose = torch.zeros(
                            [body_pose.shape[0], 6], dtype=body_pose.dtype, device=body_pose.device
                        )
                        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                    est_params_i["body_pose"] = body_pose
                elif key == "betas":
                    est_params_i[key] = torch.zeros([1, 10], dtype=dtype, device=device)
                else:
                    est_params_i[key] = torch.tensor(est_params[key][idx], dtype=dtype, device=device)

            model_output = model(**est_params_i)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()

            out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
            )
            mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
            light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
            scene.add(mesh, "mesh")

            camera_center = [128.0, 128.0]
            camera_transl = np.zeros(3)
            camera_transl[0] = camera_default_transl[0]
            camera_transl[1] = camera_default_transl[1]
            camera_transl[2] = camera_default_transl[2]

            camera_transl[2] *= -1.0
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl
            camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            # camera_pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            pose = np.eye(4)
            pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            scene.add(light, pose=pose)

            focal_length = 5000
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length, cx=camera_center[0], cy=camera_center[1]
            )
            scene.add(camera, pose=camera_pose)

            registered_keys = dict()

            light_nodes = _create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)

            r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

            color = pil_img.fromarray(color)
            color.save(f"{output_folder}/img_meanpose/" + video_id + "/" + frame_idx + "_front.png")

            ###############################################################################################3
            out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
            )
            mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
            light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
            scene.add(mesh, "mesh")
            # camera_center = [128., 128.]
            camera_transl = np.zeros(3)
            camera_transl[0] = camera_default_transl[2]
            camera_transl[1] = camera_default_transl[1]
            camera_transl[2] = camera_default_transl[0]
            camera_transl[2] *= -1.0
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl
            camera_pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            # camera_pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            pose = np.eye(4)
            pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            scene.add(light, pose=pose)

            focal_length = 5000
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length, cx=camera_center[0], cy=camera_center[1]
            )
            scene.add(camera, pose=camera_pose)

            light_nodes = _create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)

            r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

            color = pil_img.fromarray(color)
            color.save(f"{output_folder}/img_meanpose/" + video_id + "/" + frame_idx + "_side.png")

        keypoints = np.array(keypoints_all[video_id])

        cap_fps = 13

        size = (256, 768)

        out_frame = np.zeros((256, 768, 3), dtype=np.uint8)

        images_path = os.path.join("/D_data/SL/data/WLASL", "frames", video_id)
        images_paths = sorted(os.listdir(images_path))
        n_frames = len(images_paths)

        res_video_path = os.path.join(f"{output_folder}", "results_meanpose")
        os.makedirs(res_video_path, exist_ok=True)
        resulename = os.path.join(res_video_path, f"{video_id}_meanpose.mp4")

        tmp_folder = os.path.join("/tmp/", video_id)
        os.makedirs(tmp_folder, exist_ok=True)
        for frame_idx in range(1, n_frames - 1):
            if keypoints[frame_idx - 1, 10, 2] < 0.7 and keypoints[frame_idx - 1, 9, 2] < 0.7:
                continue

            file0 = os.path.join("/D_data/SL/data/WLASL", "frames", video_id, images_paths[frame_idx])
            file1 = os.path.join(f"{output_folder}/img_meanpose", video_id, f"{frame_idx:03d}_front.png")
            file2 = os.path.join(f"{output_folder}/img_meanpose", video_id, f"{frame_idx:03d}_side.png")

            img0 = cv2.imread(file0)
            img1 = cv2.imread(file1)
            img2 = cv2.imread(file2)

            out_frame[:256, :256, :] = img0[:, :, :]
            out_frame[:256, 256:512, :] = img1[:, :, :]
            out_frame[:256, 512:, :] = img2[:, :, :]

            cv2.imwrite(os.path.join(tmp_folder, f"{frame_idx:03d}.png"), out_frame)

        os.system(
            f"ffmpeg -hide_banner -loglevel error -framerate {cap_fps} -pattern_type glob -i '{tmp_folder}/*.png' -c:v libx264 -pix_fmt yuv420p -y {resulename}"
        )
