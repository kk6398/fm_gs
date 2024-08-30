import shutil
from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
# from scipy.spatial.transform import Rotation as R
from torch import Tensor
# #
# from ..misc.cropping import center_crop_intrinsics
# from ..model.model import ModelExports
# from ..model.projection import homogenize_points, sample_image_grid, unproject
from ..third_party.colmap.read_write_model import Camera, Image, read_model, write_model
# from open3d.utility import Quaternion
from scipy.spatial.transform import Rotation as R
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import os
import json
from einops import einsum, rearrange
from ..model.projection import homogenize_points, sample_image_grid, unproject
import open3d as o3d

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
           ].reshape(*batch_dim, 4)

def save_json(scene_info):
    json_cams = []
    camlist = []

    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    for id, cam in enumerate(camlist):  # cam: CameraInfo
        json_cams.append(camera_to_JSON(id, cam))
    output_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cameras.json"), 'w') as file:
        json.dump(json_cams, file)


def xyz_from_flowmap(depths, intrinsics, extrinsics, batch, num_images=-1):
    _, _, dh, dw = depths.shape  # ([1, 20, 160, 224])
    xy, _ = sample_image_grid((dh, dw), extrinsics.device)  # 生成图像网格的坐标，这些坐标用于后续的3D点云生成。
    if num_images == -1:
        bundle = zip(  # zip函数将外参、内参、深度图像和颜色图像打包在一起，以便在循环中一起处理。
            extrinsics[0],  # torch.Size ([20, 3, 3])
            intrinsics[0],  # torch.Size([20, 4, 4])       # 这里的intrinsic也应该是对应的original尺寸下的intrinsic
            depths[0],  # ([20, 160, 224])
            batch.videos[0],  # ([20, 3, 160, 224])
        )
    else:
        from math import floor
        train_view = []
        length = depths.shape[1]
        interval = floor((length - num_images) / (num_images - 1))
        for i in range(0, length):
            if i % (interval + 1) == 0:
                train_view.append(i)
        train_view[-1] = length - 1  # 强制让最后一个值为总数200
        # 提取batch, depths, intrinsics, extrinsics 的train_view
        batch_fewshot = []
        depth_fewshot = []
        intrinsics_fewshot = []
        extrinsics_fewshot = []
        for i in train_view:
            batch_fewshot.append(batch.videos[:, i])  # batch.videos: [1, 200, 3, xxx, xxx]
            depth_fewshot.append(depths[:, i])
            intrinsics_fewshot.append(intrinsics[:, i])
            extrinsics_fewshot.append(extrinsics[:, i])
        batch_fewshot = torch.cat(batch_fewshot, dim=0)
        depth_fewshot = torch.cat(depth_fewshot, dim=0)
        intrinsics_fewshot = torch.cat(intrinsics_fewshot, dim=0)
        extrinsics_fewshot = torch.cat(extrinsics_fewshot, dim=0)
        bundle = zip(  # zip函数将外参、内参、深度图像和颜色图像打包在一起，以便在循环中一起处理。
            extrinsics_fewshot,  # torch.Size ([20, 3, 3])
            intrinsics_fewshot,  # torch.Size([20, 4, 4])       # 这里的intrinsic也应该是对应的original尺寸下的intrinsic
            depth_fewshot,  # ([20, 160, 224])
            batch_fewshot,  # ([20, 3, 160, 224])
        )

    points = []  # 初始化两个空列表，用于存储转换后的3D点和对应的颜色。
    # colors = []
    for extrinsics, intrinsics, depths, rgb in bundle:  # 循环遍历之前打包的数据。
        xyz = unproject(xy, depths, intrinsics)  # 使用unproject函数将图像坐标和深度值转换为3D空间中的点(相机坐标系)
        xyz = homogenize_points(xyz)  # 将3D点同质化，即增加一个维度以便于矩阵乘法。
        xyz = einsum(extrinsics, xyz, "i j, ... j -> ... i")[..., :3]  # 将外参矩阵与同质化的3D点相乘，得到世界坐标系中的3D点，并去除同质化的维度。
        points.append(rearrange(xyz, "h w xyz -> (h w) xyz").detach().cpu().numpy())  # 将转换后的3D点和颜色分别添加到对应的列表中
        # colors.append(rearrange(rgb, "c h w -> (h w) c").detach().cpu().numpy())
    points = np.concatenate(points)  # 将所有3D点和颜色合并成一个NumPy数组
    # colors = np.concatenate(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("/data2/hkk/3dgs/flowmap/outputs/fused_point_cloud.ply", pcd)

    # 将点云的点坐标转换为 PyTorch 张量，并移到GPU上。
    fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()  # [716800,3]

    return fused_point_cloud


def center_crop_intrinsics(
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        old_shape: tuple[int, int],
        new_shape: tuple[int, int],
) -> Float[Tensor, "*batch 3 3"] | None:
    """Modify the given intrinsics to account for center cropping."""

    if intrinsics is None:
        return None

    h_old, w_old = old_shape  # 160  224
    h_new, w_new = new_shape  # (180, 240)
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_old / w_new  # fx      # * 180/160
    intrinsics[..., 1, 1] *= h_old / h_new  # fy      # * 240/224
    return intrinsics


def flowmap_2_gs(intrinsics, extrinsics, frame_paths, batch_video):
    cameras = {}
    images = {}

    camera_id = 1
    model_id = 1
    model_name = "PINHOLE"
    width = batch_video.shape[-1]  # 4032
    height = batch_video.shape[-2]  # 3024
    # print("batch_video.shape: ", batch_video.shape)        # torch.Size([1, 20, 3, 3024, 4032])

    # 内参应该现有(160,224) → (180,240)
    # _, _, h_cropped, w_cropped = depths.shape
    # h_uncropped, w_uncropped = uncropped_exports_shape

    # intrinsics = center_crop_intrinsics(
    #     intrinsics,
    #     (h_cropped, w_cropped),         # 160  224
    #     (h_uncropped, w_uncropped),     # (180, 240)
    # )

    # params = np.array([intrinsics[0, 0, 0, 0], intrinsics[0, 0, 1, 1], intrinsics[0, 0, 0, 2], intrinsics[0, 0, 1, 2]])
    # print("intrinsics: ", intrinsics)                # 归一化后的参数

    for index, k in enumerate(intrinsics):              # 按索引提取每张图像对应的内参    intrinsics: [20,3,3]
        id = index + 1
        camera_id = 1
        # Undo the normalization we apply to the intrinsics.
        k = k.detach().clone()
        k = k.cpu().numpy()
        k[0] *= width              # 将k矩阵的第一行的每个元素乘以变量w，第二行的每个元素乘以变量h
        k[1] *= height              # 为了撤销之前对内参进行的归一化处理

        # Extract the intrinsics' parameters.
        fx = k[0, 0]                         # fx和fy分别是沿着图像宽度和高度的焦距，cx和cy分别是图像的主点坐标
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]
        params = [fx,fy,cx,cy]
        cameras[id] = Camera(id, "PINHOLE", width, height, np.array(params))

    # params_cpu = intrinsics.cpu().detach().numpy()  # [1,20,3,3]
    # params = [params_cpu[0, 0, 0, 0]*width, params_cpu[0, 0, 1, 1]*height, params_cpu[0, 0, 0, 2]*width, params_cpu[0, 0, 1, 2]*height]
    # # params = [params_cpu[0, 0, 0, 0], params_cpu[0, 0, 1, 1], params_cpu[0, 0, 0, 2], params_cpu[0, 0, 1, 2]]
    # # print("params: ", params)               # params:   [0.7121296, 0.99698144, 0.5, 0.5]
    # cameras[camera_id] = Camera(id=camera_id,  # 内参信息
    #                             model=model_name,
    #                             width=width,
    #                             height=height,
    #                             params=params)
    # print("cameras: ", cameras)

    # for i in range(extrinsics.shape[1]):
    #     extrinsic_matrix = extrinsics[:, i, :, :]     # 外参也初始化了为单位矩阵 #
    #     extrinsic_matrix = extrinsic_matrix.squeeze(0)
    #     # print("extrinsic_matrix: ", extrinsic_matrix.shape)   # torch.Size([4, 4])
    #     image_id = i + 1
    #     R = extrinsic_matrix[:3, :3]      # 3*3的单位矩阵
    #     t = extrinsic_matrix[:3, 3]       # 1*3的零矩阵
    #     qvec = matrix_to_quaternion(R).tolist()    # 转换为四元数  [1.0, 0.0, 0.0, 0.0]
    #     tvec = t
    #     camera_id = 1
    #     image_name = frame_paths[i].name     # 1: PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG')
    #
    #     images[image_id] = Image(                               # 外参信息
    #         id=image_id, qvec=qvec, tvec=tvec,
    #         camera_id=camera_id, name=image_name, xys=[], point3D_ids=[])

    for i in range(extrinsics.shape[0]):
        extrinsic_matrix = extrinsics[i, :, :]  # [4,4]
        image_id = i + 1
        # w2c = extrinsic_matrix.detach().cpu().numpy()  # 此时外参是C2W, 需要转换乘W2C
        w2c = extrinsic_matrix.inverse().detach().cpu().numpy()
        qx, qy, qz, qw = R.from_matrix(w2c[:3, :3]).as_quat()  # R: Rotation
        qvec = np.array((qw, qx, qy, qz))
        # qvec = R.from_matrix(w2c[:3, :3]).as_quat()  # R: Rotation
        tvec = w2c[:3, 3]
        camera_id = 1
        image_name = frame_paths[i].name  # 1: PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG')

        # R = extrinsic_matrix[:3, :3]  # 3*3的单位矩阵
        # t = extrinsic_matrix[:3, 3]  # 1*3的零矩阵
        # qvec = matrix_to_quaternion(R).tolist()  # 转换为四元数  [1.0, 0.0, 0.0, 0.0]   # 考虑不转换
        # tvec = t
        # camera_id = 1
        # image_name = frame_paths[i].name  # 1: PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG')

        images[image_id] = Image(  # 外参信息
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=image_name, xys=[], point3D_ids=[])

    return images, cameras  # 分别对应  外参和内参