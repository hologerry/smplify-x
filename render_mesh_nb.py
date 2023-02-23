import os
import os.path as osp
import pickle
import sys

from collections import defaultdict

import cv2
import numpy as np
import PIL.Image as pil_img
import pyrender
import torch
import trimesh

from tqdm import tqdm


os.environ["QT_QPA_PLATFORM"] = "offscreen"


out_mesh = trimesh.load_mesh("smplx_debug/meshes/00583/001.obj")
img = cv2.imread("/D_data/SL/data/WLASL/WLASL2000_frames/00583/0001.png")


class MeshViewer(object):
    def __init__(self, width=1200, height=800, body_color=(1.0, 1.0, 0.9, 1.0), registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        import pyrender
        import trimesh

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.body_color = body_color
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3])
        self.scene.add(pc, pose=camera_pose)

        self.viewer = pyrender.Viewer(
            self.scene,
            use_raymond_lighting=True,
            viewport_size=(width, height),
            cull_faces=False,
            run_in_thread=True,
            registered_keys=registered_keys,
        )

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0), wireframe=False):

        material = self.mat_constructor(metallicFactor=0.0, alphaMode="BLEND", baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces):
        if not self.viewer.is_active:
            return

        self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name == "body_mesh":
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(vertices, faces, color=self.body_color)
        self.scene.add(body_mesh, name="body_mesh")

        self.viewer.render_lock.release()


out_img_fn = "overlay.png"
material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
)
mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
scene.add(mesh, "mesh")

camera_center = np.array([960.0, 540.0], dtype=np.float32)

camera_transl = np.array([0.00559422, 0.4709179, 4.6253934], dtype=np.float32)

focal_length = 1500.0

W = 1920
H = 1080


# Equivalent to 180 degrees around the y-axis. Transforms the fit to
# OpenGL compatible coordinate system.
camera_transl[0] *= -1.0

camera_pose = np.eye(4)
camera_pose[:3, 3] = camera_transl

camera = pyrender.camera.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=camera_center[0], cy=camera_center[1])
scene.add(camera, pose=camera_pose)

mv = MeshViewer(body_color=(1.0, 1.0, 0.9, 1.0))

# Add lights to the mesh viewer
light_nodes = mv.viewer._create_raymond_lights()

for node in light_nodes:
    scene.add_node(node)

r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
color = color.astype(np.float32) / 255.0

valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
input_img = img
output_img = color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img

img = pil_img.fromarray((output_img * 255).astype(np.uint8))
img.save(out_img_fn)
