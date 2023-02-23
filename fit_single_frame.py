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


import math
import os
import os.path as osp
import pickle
import sys
import time

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_img
import torch

from human_body_prior.tools.model_loader import load_vposer
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

import fitting

from optimizers import optim_factory


def imshow_keypoints(
    img,
    pose_result,
    skeleton=None,
    kpt_score_thr=0.3,
    pose_kpt_color=None,
    pose_link_color=None,
    radius=2,
    thickness=1,
    show_keypoint_weight=False,
):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    # img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [15, 17],
        [15, 18],
        [15, 19],
        [16, 20],
        [16, 21],
        [16, 22],
        [91, 92],
        [92, 93],
        [93, 94],
        [94, 95],
        [91, 96],
        [96, 97],
        [97, 98],
        [98, 99],
        [91, 100],
        [100, 101],
        [101, 102],
        [102, 103],
        [91, 104],
        [104, 105],
        [105, 106],
        [106, 107],
        [91, 108],
        [108, 109],
        [109, 110],
        [110, 111],
        [112, 113],
        [113, 114],
        [114, 115],
        [115, 116],
        [112, 117],
        [117, 118],
        [118, 119],
        [119, 120],
        [112, 121],
        [121, 122],
        [122, 123],
        [123, 124],
        [112, 125],
        [125, 126],
        [126, 127],
        [127, 128],
        [112, 129],
        [129, 130],
        [130, 131],
        [131, 132],
    ]

    palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ]
    )

    pose_link_color = palette[
        [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
        + [16, 16, 16, 16, 16, 16]
        + [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]
        + [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]
    ]

    pose_kpt_color = palette[
        [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)
    ]

    for ii in range(1):

        kpts = np.array(pose_result, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius, color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(img_copy, transparency, img, 1 - transparency, 0, dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (
                    pos1[0] <= 0
                    or pos1[0] >= img_w
                    or pos1[1] <= 0
                    or pos1[1] >= img_h
                    or pos2[0] <= 0
                    or pos2[0] >= img_w
                    or pos2[1] <= 0
                    or pos2[1] >= img_h
                    or kpts[sk[0], 2] < kpt_score_thr
                    or kpts[sk[1], 2] < kpt_score_thr
                    or pose_link_color[sk_id] is None
                ):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(img_copy, transparency, img, 1 - transparency, 0, dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


def hard_set_keypoints(keypoints):
    len_shoulder = keypoints[0, 5, 0] - keypoints[0, 2, 0]
    len_waist = len_shoulder / 1.7
    len_nose_to_shoulder = keypoints[0, 0, 1] - keypoints[0, 1, 1]

    keypoints[:, 8, 0] = keypoints[:, 1, 0]
    keypoints[:, 8, 1] = keypoints[:, 1, 1] + 1.5 * len_shoulder
    keypoints[:, 8, 2] = 0.65
    keypoints[:, 9, 0] = keypoints[:, 8, 0] - 0.5 * len_waist
    keypoints[:, 9, 1] = keypoints[:, 8, 1]
    keypoints[:, 9, 2] = 0.65
    keypoints[:, 12, 0] = keypoints[:, 8, 0] + 0.5 * len_waist
    keypoints[:, 12, 1] = keypoints[:, 8, 1]
    keypoints[:, 12, 2] = 0.65

    keypoints[:, 10, 0] = keypoints[0, 9, 0]
    keypoints[:, 10, 1] = keypoints[0, 9, 1] + 2.0 * len_waist
    keypoints[:, 10, 2] = 0.65
    keypoints[:, 11, 0] = keypoints[0, 9, 0]
    keypoints[:, 11, 1] = keypoints[0, 9, 1] + 4.0 * len_waist
    keypoints[:, 11, 2] = 0.65
    keypoints[:, 13, 0] = keypoints[0, 12, 0]
    keypoints[:, 13, 1] = keypoints[0, 12, 1] + 2.0 * len_waist
    keypoints[:, 13, 2] = 0.65
    keypoints[:, 14, 0] = keypoints[0, 12, 0]
    keypoints[:, 14, 1] = keypoints[0, 12, 1] + 4.0 * len_waist
    keypoints[:, 14, 2] = 0.65

    return keypoints


def fit_single_frame(
    img,
    idx,
    keypoints,
    keypoints_render,
    body_model,
    camera,
    joint_weights,
    body_pose_prior,
    jaw_prior,
    left_hand_prior,
    right_hand_prior,
    shape_prior,
    expr_prior,
    angle_prior,
    joints_blur=None,
    joints_fix=None,
    camera_transl=None,
    camera_orient=None,
    betas_fix=None,
    result_fn="out.pkl",
    mesh_fn="out.obj",
    loss_type="smplify",
    use_cuda=True,
    init_joints_idxs=(9, 12, 2, 5),
    use_face=True,
    use_hands=True,
    data_weights=None,
    body_pose_prior_weights=None,
    hand_pose_prior_weights=None,
    jaw_pose_prior_weights=None,
    shape_weights=None,
    expr_weights=None,
    hand_joints_weights=None,
    face_joints_weights=None,
    depth_loss_weight=1e2,
    interpenetration=True,
    coll_loss_weights=None,
    df_cone_height=0.5,
    penalize_outside=True,
    max_collisions=8,
    point2plane=False,
    part_segm_fn="",
    focal_length=5000.0,
    side_view_thsh=25.0,
    rho=100,
    vposer_latent_dim=32,
    vposer_ckpt="",
    use_joints_conf=False,
    interactive=True,
    visualize=False,
    save_meshes=True,
    degrees=None,
    batch_size=1,
    dtype=torch.float32,
    ign_part_pairs=None,
    left_shoulder_idx=2,
    right_shoulder_idx=5,
    **kwargs
):
    assert batch_size == 1, "PyTorch L-BFGS only supports batch_size == 1"
    print(idx)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [
            1,
        ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = "Number of Body pose prior weights {}".format(
        len(body_pose_prior_weights)
    ) + " does not match the number of data term weights {}".format(len(data_weights))
    assert len(data_weights) == len(body_pose_prior_weights), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, 0.5 * 1e1]
        msg = "Number of Body pose prior weights does not match the" + " number of hand pose prior weights"
        assert len(hand_pose_prior_weights) == len(body_pose_prior_weights), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = "Number of Body pose prior weights does not match the" + " number of hand joint distance weights"
            assert len(hand_joints_weights) == len(body_pose_prior_weights), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, 0.5 * 1e1]
    msg = "Number of Body pose prior weights = {} does not match the" + " number of Shape prior weights = {}"
    assert len(shape_weights) == len(body_pose_prior_weights), msg.format(
        len(shape_weights), len(body_pose_prior_weights)
    )

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(",")), jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = "Number of Body pose prior weights does not match the" + " number of jaw pose prior weights"
        assert len(jaw_pose_prior_weights) == len(body_pose_prior_weights), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, 0.5 * 1e1]
        msg = "Number of Body pose prior weights = {} does not match the" + " number of Expression prior weights = {}"
        assert len(expr_weights) == len(body_pose_prior_weights), msg.format(
            len(body_pose_prior_weights), len(expr_weights)
        )

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = "Number of Body pose prior weights does not match the" + " number of face joint distance weights"
        assert len(face_joints_weights) == len(body_pose_prior_weights), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = "Number of Body pose prior weights does not match the" + " number of collision loss weights"
    assert len(coll_loss_weights) == len(body_pose_prior_weights), msg

    use_vposer = kwargs.get("use_vposer", True)
    vposer, pose_embedding = [
        None,
    ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32], dtype=dtype, device=device, requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model="snapshot")
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim], dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoints = hard_set_keypoints(keypoints)

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    visibility = keypoint_data[:, :, 2]

    # print(gt_joints)
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    visibility = visibility.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        import mesh_intersection.loss as collisions_loss

        from mesh_intersection.bvh_search_tree import BVH
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, "Interpenetration term can only be used with CUDA"
        assert torch.cuda.is_available(), "No CUDA Device! Interpenetration term can only be used" + " with CUDA"

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
            sigma=df_cone_height, point2plane=point2plane, vectorized=True, penalize_outside=penalize_outside
        )

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, "rb") as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file, encoding="latin1")
            faces_segm = face_segm_data["segm"]
            faces_parents = face_segm_data["parents"]
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents, ign_part_pairs=ign_part_pairs
            ).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {
        "data_weight": data_weights,
        "body_pose_weight": body_pose_prior_weights,
        "shape_weight": shape_weights,
    }
    if use_face:
        opt_weights_dict["face_weight"] = face_joints_weights
        opt_weights_dict["expr_prior_weight"] = expr_weights
        opt_weights_dict["jaw_prior_weight"] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict["hand_weight"] = hand_joints_weights
        opt_weights_dict["hand_prior_weight"] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict["coll_loss_weight"] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [
        dict(zip(keys, vals)) for vals in zip(*(opt_weights_dict[k] for k in keys if opt_weights_dict[k] is not None))
    ]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key], device=device, dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get("body_tri_idxs")
    init_t = fitting.guess_init(
        body_model,
        gt_joints,
        edge_indices,
        use_vposer=use_vposer,
        vposer=vposer,
        pose_embedding=pose_embedding,
        model_type=kwargs.get("model_type", "smpl"),
        focal_length=focal_length,
        dtype=dtype,
    )
    # print(init_t)
    camera_loss = fitting.create_loss(
        "camera_init",
        trans_estimation=init_t,
        init_joints_idxs=init_joints_idxs,
        depth_loss_weight=depth_loss_weight,
        dtype=dtype,
    ).to(device=device)
    camera_loss.trans_estimation[:] = init_t
    #  print(init_t)
    loss = fitting.create_loss(
        loss_type=loss_type,
        joint_weights=joint_weights,
        rho=rho,
        use_joints_conf=use_joints_conf,
        use_face=use_face,
        use_hands=use_hands,
        vposer=vposer,
        pose_embedding=pose_embedding,
        body_pose_prior=body_pose_prior,
        shape_prior=shape_prior,
        angle_prior=angle_prior,
        expr_prior=expr_prior,
        left_hand_prior=left_hand_prior,
        right_hand_prior=right_hand_prior,
        jaw_prior=jaw_prior,
        interpenetration=interpenetration,
        pen_distance=pen_distance,
        search_tree=search_tree,
        tri_filtering_module=filter_faces,
        dtype=dtype,
        **kwargs
    )
    loss = loss.to(device=device)
    #   camera=camera.to(device=device)
    with fitting.FittingMonitor(batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H
        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({"data_weight": data_weight})

        # Reset the parameters to estimate the initial translation of the
        # body model
        body_model.reset_params(body_pose=body_mean_pose)

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx], gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        camera.translation.requires_grad = True
        #  camera.translation=torch.tensor([-4.9302e-03, 7.1391e-01, 2.0452e+01])
        # Re-enable gradient calculation for the camera translation

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(camera_opt_params, **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer,
            body_model,
            camera,
            gt_joints,
            visibility,
            joints_blur,
            idx,
            joints_fix,
            camera_loss,
            create_graph=camera_create_graph,
            use_vposer=use_vposer,
            vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False,
            return_verts=False,
        )

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(
            camera_optimizer,
            fit_camera,
            camera_opt_params,
            body_model,
            use_vposer=use_vposer,
            pose_embedding=pose_embedding,
            vposer=vposer,
        )

        if idx != 0:
            camera_opt_params[0].requires_grad = False
            camera_opt_params[0][0] = camera_transl
            camera_opt_params[1].requires_grad = False
            camera_opt_params[1][0] = camera_orient

        else:
            camera_opt_params[0].requires_grad = True
            camera_opt_params[1].requires_grad = True

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        #          tqdm.write('Camera initialization done after {:.4f}'.format(
        #               time.time() - camera_init_start))
        #          tqdm.write('Camera initialization final loss {:.4f}'.format(
        #              cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(cv2.Rodrigues(np.array([0.0, np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient, dtype=dtype, device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0

        for or_idx, orient in enumerate(tqdm(orientations, desc="Orientation")):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient, body_pose=body_mean_pose)
            body_model.reset_params(**new_params)
            if idx != 0:
                body_model.betas.requires_grad = False
                body_model.betas[:, :10] = betas_fix[:10]

            else:
                body_model.betas.requires_grad = True

            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc="Stage")):

                body_params = list(body_model.parameters())

                final_params = list(filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)
                body_optimizer.zero_grad()

                curr_weights["data_weight"] = data_weight
                curr_weights["bending_prior_weight"] = 3.17 * curr_weights["body_pose_weight"]
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights["hand_weight"]
                if use_face:
                    joint_weights[:, 67:] = curr_weights["face_weight"]
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer,
                    body_model,
                    camera=camera,
                    gt_joints=gt_joints,
                    visibility=visibility,
                    joints_blur=joints_blur,
                    idx=idx,
                    joints_fix=joints_fix,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss,
                    create_graph=body_create_graph,
                    use_vposer=use_vposer,
                    vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True,
                    return_full_pose=True,
                )

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure,
                    final_params,
                    body_model,
                    pose_embedding=pose_embedding,
                    vposer=vposer,
                    use_vposer=use_vposer,
                )

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write("Stage {:03d} done after {:.4f} seconds".format(opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write("Body fitting Orientation {} done after {:.4f} seconds".format(or_idx, elapsed))
                # tqdm.write('Body final loss val = {:.5f}'.format(final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {"camera_" + str(key): val.detach().cpu().numpy() for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy() for key, val in body_model.named_parameters()})
            if use_vposer:
                result["body_pose"] = pose_embedding.detach().cpu().numpy()
            results.append({"result": result})

    if save_meshes:
        print("Saving results and meshes")
        body_pose = vposer.decode(pose_embedding, output_type="aa").view(1, -1) if use_vposer else None

        model_type = kwargs.get("model_type", "smpl")
        append_wrists = model_type == "smpl" and use_vposer
        if append_wrists:
            wrist_pose = torch.zeros([body_pose.shape[0], 6], dtype=body_pose.dtype, device=body_pose.device)
            body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        joints_blur = model_output.body_pose.reshape(1, 21, 3).detach()
        fix_points_idx = torch.tensor([0, 1, 2, 5, 8, 9, 10, 12, 13], dtype=torch.long).to(device=joints_blur.device)
        joints_fix = torch.index_select(joints_blur, 1, fix_points_idx)[0, :, :]
        camera_transl_1 = camera.translation[0, :].detach()
        camera_orient = model_output.global_orient[0, :].detach()
        betas_fix = model_output.betas[0].detach()

        results.append(
            {
                "body_pose_rot": model_output.body_pose.detach().cpu().numpy(),
                "left_hand_pose_rot": model_output.left_hand_pose.detach().cpu().numpy(),
                "right_hand_pose_rot": model_output.right_hand_pose.detach().cpu().numpy(),
            }
        )

        with open(result_fn, "wb") as result_file:
            pickle.dump(results, result_file, protocol=2)
        print(f"Saved results to {result_fn}")

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # print(body_model.output)
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)
        print(f"Saved mesh to {mesh_fn}")

    nan_flag = False
    if final_loss_val is None:
        nan_flag = True
        print("nan_flag:true")
    return joints_blur, joints_fix, camera_transl_1, camera_orient, betas_fix, nan_flag
