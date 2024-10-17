#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# from ..scene.cameras import Camera, DifferentiableCamera, DifferentiableCamera_eval
from ..scene.cameras import Camera, DifferentiableCamera_eval
import numpy as np
from .general_utils import PILtoTorch
from .graphics_utils import fov2focal
from ..scene.dataset_readers import CameraInfo
import torch
from jaxtyping import Float
from torch import Tensor, nn
from einops import rearrange

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    resolution_args = -1
    if resolution_args in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * resolution_args)), round(
            orig_h / (resolution_scale * resolution_args))
    else:  # should be a type that converts to float
        if resolution_args == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution_args

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))  # 1600，1200

    resized_image_rgb = PILtoTorch(cam_info.image,
                                   resolution)  # 3x1200x1600      # 将original图像resize为裁剪尺寸，并读取    # 1:418

    gt_image = resized_image_rgb[:3,
               ...]  # start被省略了，表示从数组的起始位置开始切片，stop被设置为3，表示切片结束位置为索引为2的元素（不包含在切片中），step也被省略了，表示使用默认的步长1。
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask, args=args,
                  image_name=cam_info.image_name, uid=id, data_device="cuda")

def loadCam_eval(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    resolution_args = -1
    if resolution_args in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * resolution_args)), round(
            orig_h / (resolution_scale * resolution_args))
    else:  # should be a type that converts to float
        if resolution_args == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution_args

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))  # 1600，1200

    resized_image_rgb = PILtoTorch(cam_info.image,
                                   resolution)  # 3x1200x1600      # 将original图像resize为裁剪尺寸，并读取    # 1:418

    gt_image = resized_image_rgb[:3,...]  # start被省略了，表示从数组的起始位置开始切片，stop被设置为3，表示切片结束位置为索引为2的元素（不包含在切片中），step也被省略了，表示使用默认的步长1。
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if args.testeval == 1:
        cam = DifferentiableCamera_eval(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                      image=gt_image, gt_alpha_mask=loaded_mask, args=args,
                      image_name=cam_info.image_name, uid=id, data_device="cuda").requires_grad_(False).to(args.data_device)
        return cam
    else:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,       # DifferentiableCamera
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                      image=gt_image, gt_alpha_mask=loaded_mask, args=args,
                      image_name=cam_info.image_name, uid=id, data_device="cuda")

    # cam = DifferentiableCamera(
    #     uid=cam_info.uid,
    #     image_name=cam_info.image_name,
    #     load_device=args.data_device,
    #
    #     width=cam_info.width,
    #     height=cam_info.height,
    #
    #     image_width=resolution[0],
    #     image_height=resolution[1],
    #
    #     quaternionOrR=torch.eye(3, dtype=torch.float32, device=args.data_device),       # 初始化R 和 T
    #     T=torch.zeros(3, dtype=torch.float32, device=args.data_device),
    #
    #     FoVx=cam_info.FovX,
    #     FoVy=cam_info.FovY,
    # ).requires_grad_(False).to(args.data_device)
    #
    # return cam


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []  # validation_configs时，先读取所有相机视角的参数，再选取5个

    for id, c in enumerate(cam_infos):
        # camera_list.append(loadCam_eval(args, id, c, resolution_scale))
        camera_list.append(loadCam(args, id, c, resolution_scale))

    # if args.testeval == 1:
    #     for id, c in enumerate(cam_infos):
    #         camera_list.append(loadCam_eval(args, id, c, resolution_scale))
    # else:
    #     for id, c in enumerate(cam_infos):
    #         camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_from_camInfos_selection(id, cam_infos, resolution_scale, args):
    # 只有一个视角的相机参数
    return loadCam(args, id, cam_infos[id], resolution_scale)


# def camera_to_JSON(id, camera : Camera):
def camera_to_JSON(id, camera: CameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    # i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
    # return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

def pose_to_extrinsics(pose):     # pose: [3,7]
    quaternion = pose[:, :4]    #   [3,4]
    translation = pose[:, 4:]   #  [3,3]
    extr_rotation = quaternion_to_matrix(quaternion)
    extr_tranlation = translation
    extrinsics = torch.eye(4).repeat(pose.shape[0], 1, 1)
    extrinsics[:, :3, :3] = extr_rotation
    extrinsics[:, :3, 3] = extr_tranlation

    return extrinsics
