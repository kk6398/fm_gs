#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
###

import torch
from torch import nn
import numpy as np
from ..utils.graphics_utils import getWorld2View2, getProjectionMatrix
from torch.nn import functional as F

# class Camera(nn.Module):
#     def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
#                  image_name, uid, args,
#                  trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
#                  ):
#         super(Camera, self).__init__()
#
#         self.uid = uid
#         self.colmap_id = colmap_id
#         self.R = R         # original: float64,  modification: float32
#         self.T = T         # float32
#         self.FoVx = FoVx
#         self.FoVy = FoVy
#         self.image_name = image_name
#         self.args = args
#         try:
#             self.data_device = torch.device(data_device)
#         except Exception as e:
#             print(e)
#             print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
#             self.data_device = torch.device("cuda")
#
#         self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
#         self.image_width = self.original_image.shape[2]
#         self.image_height = self.original_image.shape[1]
#
#         if gt_alpha_mask is not None:
#             self.original_image *= gt_alpha_mask.to(self.data_device)
#         else:
#             self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
#
#         self.zfar = 100.0
#         self.znear = 0.01
#
#         self.trans = trans
#         self.scale = scale
#
#         self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, args,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R         # original: float64,  modification: float32
        self.T = T         # float32
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.args = args
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # R:matrix T:
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    #     register = lambda x, y: self.register_parameter(x, torch.nn.Parameter(y))
    #     assert R.dtype == T.dtype
    #
    #     if args.testeval == 1:
    #         quaternionOrR = torch.eye(3, dtype=torch.float32, device=data_device)
    #         T = torch.zeros(3, dtype=torch.float32, device=data_device)
    #     else:
    #         quaternionOrR = R.reshape(-1)  # R 3x3
    #         quaternionOrR = torch.tensor(quaternionOrR, dtype=torch.float32, device=self.data_device)
    #         T = torch.tensor(T, dtype=torch.float32, device=self.data_device)
    #
    #     # quaternionOrR = R.reshape(-1)  # R 3x3
    #     # quaternionOrR = torch.tensor(quaternionOrR, dtype=torch.float32, device=self.data_device)
    #     # T = torch.tensor(T, dtype=torch.float32, device=self.data_device)
    #
    #     _quaternion = quaternionOrR.reshape(4, ) if len(quaternionOrR) == 4 else self.matrix_to_quaternion(
    #         quaternionOrR.reshape(3, 3)).reshape(4, )      # float32
    #     _T = T.reshape(3, )
    #     register('quaternion', _quaternion)  # 使其可微
    #     register('T', _T)
    #
    # def __getattr__(self, name: str):
    #     if '_parameters' in self.__dict__:
    #         _parameters = self.__dict__['_parameters']
    #         if name in _parameters:
    #             return _parameters[name]
    #     if '_buffers' in self.__dict__:
    #         _buffers = self.__dict__['_buffers']
    #         if name in _buffers:
    #             return _buffers[name]
    #     if '_modules' in self.__dict__:
    #         modules = self.__dict__['_modules']
    #         if name in modules:
    #             return modules[name]
    #     return self.__getattribute__(name)
    #
    # def cam_requires_grad_(self, requires_grad=False):
    #     self.quaternion.requires_grad_(requires_grad)
    #     self.T.requires_grad_(requires_grad)
    #
    # def init_(self, cam):
    #     self.quaternion.data.copy_(cam.quaternion.data)
    #     self.T.data.copy_(cam.T.data)
    #     return self
    #
    # def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    #     r, i, j, k = torch.unbind(quaternions, -1)
    #     two_s = 2.0 / (quaternions * quaternions).sum(-1)
    #     o = torch.stack(
    #         (
    #             1 - two_s * (j * j + k * k),
    #             two_s * (i * j - k * r),
    #             two_s * (i * k + j * r),
    #             two_s * (i * j + k * r),
    #             1 - two_s * (i * i + k * k),
    #             two_s * (j * k - i * r),
    #             two_s * (i * k - j * r),
    #             two_s * (j * k + i * r),
    #             1 - two_s * (i * i + j * j),
    #         ),
    #         -1,
    #     )
    #     return o.reshape(quaternions.shape[:-1] + (3, 3))
    #
    # def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
    #     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    #         raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    #
    #     def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    #         ret = torch.zeros_like(x)
    #         positive_mask = x > 0
    #         ret[positive_mask] = torch.sqrt(x[positive_mask])
    #         return ret
    #
    #     batch_dim = matrix.shape[:-2]
    #     m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
    #         matrix.reshape(batch_dim + (9,)), dim=-1
    #     )
    #
    #     q_abs = _sqrt_positive_part(
    #         torch.stack(
    #             [
    #                 1.0 + m00 + m11 + m22,
    #                 1.0 + m00 - m11 - m22,
    #                 1.0 - m00 + m11 - m22,
    #                 1.0 - m00 - m11 + m22,
    #             ],
    #             dim=-1,
    #         )
    #     )
    #
    #     quat_by_rijk = torch.stack(
    #         [
    #             torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
    #             torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
    #             torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
    #             torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    #         ],
    #         dim=-2,
    #     )
    #
    #     flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    #     quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    #
    #     return quat_candidates[
    #            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    #            ].reshape(batch_dim + (4,))
    #
    # def R(self) -> torch.Tensor:
    #     return self.quaternion_to_matrix(self.quaternion)


class DifferentiableCamera_eval(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, args,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
                 ):
        super(DifferentiableCamera_eval, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R_raw = R  # original: float64,  modification: float32
        self.t = T  # float32
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.args = args
        # self.projection_matrix = torch.empty(0)
        # self.full_proj_transform = torch.empty(0)
        # self.camera_center = torch.empty(0)


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # R:matrix [1,0,0] [0,1,0] [0,0,1]      T: [0,0,0]
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

        register = lambda x, y: self.register_parameter(x, torch.nn.Parameter(y))
        assert R.dtype == T.dtype

        if args.testeval == 1:
            quaternionOrR = torch.eye(3, dtype=torch.float32, device=data_device)
            T = torch.zeros(3, dtype=torch.float32, device=data_device)
        else:
            quaternionOrR = R.reshape(-1)  # R 3x3
            quaternionOrR = torch.tensor(quaternionOrR, dtype=torch.float32, device=self.data_device)
            T = torch.tensor(T, dtype=torch.float32, device=self.data_device)

        _quaternion = quaternionOrR.reshape(4, ) if len(quaternionOrR) == 4 else self.matrix_to_quaternion(
            quaternionOrR.reshape(3, 3)).reshape(4, )  # float32
        _T = T.reshape(3, )

        # _quaternion = R
        # _quaternion = torch.from_numpy(R).cuda()
        # _T = T
        # _T = torch.from_numpy(T).cuda()

        self.quaternion = nn.Parameter(_quaternion.requires_grad_(True))
        self.T = nn.Parameter(_T.requires_grad_(True))
        register('quaternion', _quaternion)  # 使其可微  # self.quaternion
        register('T', _T)

        # self.R = self.quaternion_to_matrix(self.quaternion)

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return self.__getattribute__(name)
    #
    def cam_requires_grad_(self, requires_grad=False):
        self.quaternion.requires_grad_(requires_grad)
        self.T.requires_grad_(requires_grad)
    #
    #

    def init_(self, cam):
        self.quaternion.data.copy_(cam.quaternion.data)
        self.T.data.copy_(cam.T.data)
        return self

    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
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


    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
            ret = torch.zeros_like(x)
            positive_mask = x > 0
            ret[positive_mask] = torch.sqrt(x[positive_mask])
            return ret

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
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

        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        return quat_candidates[
               F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
               ].reshape(batch_dim + (4,))

    # def R(self) -> torch.Tensor:
    #     # b = self.quaternion
    #     # print("b:", b)
    #     # c = self.quaternion_to_matrix(b)
    #     return self.quaternion_to_matrix(self.quaternion)

    @property
    def R(self):
        return self.quaternion_to_matrix(self.quaternion)

    @property
    def world_view_transform(self):
        matrix = torch.eye(4, device=self.quaternion.device, dtype=self.quaternion.dtype, requires_grad=False)
        matrix[:3, :3] = (self.R).T
        matrix[:3, 3] = self.T
        return matrix.T

    # def world_view_transform(self) -> torch.Tensor:
    #     matrix = torch.eye(4, device=self.quaternion.device, dtype=self.quaternion.dtype, requires_grad=False)
    #     # matrix = torch.eye(4, device=self.quaternion.device, dtype=self.quaternion.dtype, requires_grad=True)
    #     # a = self.R()
    #     # matrix[:3, :3] = (self.R).T
    #     self.RR = self.quaternion_to_matrix(self.quaternion)
    #     matrix[:3, :3] = (self.RR).T
    #     # matrix[:3, :3] = (self.R()).T
    #     matrix[:3, 3] = self.T
    #     return matrix.T
        #
        # new_matrix = matrix.clone()
        # new_matrix[:3, :3] = (self.R()).T
        # new_matrix[:3, 3] = self.T
        # return new_matrix.T

    @property
    # def projection_matrix(self):
    #     return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()

    @property
    def full_proj_transform(self) -> torch.Tensor:
        return self.world_view_transform @ self.projection_matrix

    # def full_proj_transform(self):
    #     return (self.world_view_transform) * (self.projection_matrix)

    @property
    def camera_center(self) -> torch.Tensor:
        return -self.T.view(1, 3) @ self.R.T
        # return -self.T.view(1, 3) @ self.RR.T

    # def R(self) -> torch.Tensor:
        #     return self.quaternion_to_matrix(self.quaternion)

#
# class DifferentiableCamera(nn.Module):
#     def __init__(self,
#         # Property
#         uid,
#         image_name,
#         load_device,
#         width,
#         height,
#         image_width,
#         image_height,
#         # image_path: str, depth_path: str, seg_mask_path: str,
#         # Extrinsics (init.)
#         quaternionOrR, T,
#         # Intrinsics
#         FoVx, FoVy,
#
#         # Offsetx: float, Offsety: float,
#         # # Options
#         # scale_and_shift_mode: str,
#         # # Reference Extrinsics (facilitate evaluation)
#         # ref_quaternionOrR: torch.Tensor = None, ref_T: torch.Tensor = None,
#     ) -> None:
#         super().__init__()
#         # super(DifferentiableCamera, self).__init__()
#         self.uid = uid
#         # print("1111111111111111 ", self.uid)
#         self.image_name = image_name
#         self.load_device = load_device
#         self.width, self.height = width, height
#         self.image_width, self.image_height = image_width, image_height
#         self.registered = False
#         self.optimized_iteration = 0
#
#         # self.image_path = image_path
#
#         register = lambda x, y: self.register_parameter(x, torch.nn.Parameter(y))
#         # Extrinsics
#         assert quaternionOrR.dtype == T.dtype
#         # print("quaternionOrR: ", quaternionOrR)      # tensor 3x3  identity
#         quaternionOrR = quaternionOrR.reshape(-1)    # tensor([1., 0., 0., 0., 1., 0., 0., 0., 1.], device='cuda:0')
#         _quaternion = quaternionOrR.reshape(4, ) if len(quaternionOrR) == 4 else self.matrix_to_quaternion(quaternionOrR.reshape(3, 3)).reshape(4, )               # 判断表示形式：R 还是quater四元数
#         print("_quaternion:", _quaternion)           # _quaternion: tensor([1., 0., 0., 0.]
#         _T = T.reshape(3, )                          # _T: ([0., 0., 0.]
#         print("_T:", _T)
#
#         register('quaternion', _quaternion)      # 使其可微
#         register('T', _T)
#
#     # # def __getattr__(self, name: str):
#     # #     if '_parameters' in self.__dict__:
#     # #         _parameters = self.__dict__['_parameters']
#     # #         if name in _parameters:
#     # #             return _parameters[name]
#     # #     if '_buffers' in self.__dict__:
#     # #         _buffers = self.__dict__['_buffers']
#     # #         if name in _buffers:
#     # #             return _buffers[name]
#     # #     if '_modules' in self.__dict__:
#     # #         modules = self.__dict__['_modules']
#     # #         if name in modules:
#     # #             return modules[name]
#     # #     return self.__getattribute__(name)
#
#     def cam_requires_grad_(self, requires_grad=False):
#         self.quaternion.requires_grad_(requires_grad)
#         self.T.requires_grad_(requires_grad)
#
#     def init_(self, cam):
#         self.quaternion.data.copy_(cam.quaternion.data)
#         self.T.data.copy_(cam.T.data)
#         return self
#
#     def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
#         r, i, j, k = torch.unbind(quaternions, -1)
#         two_s = 2.0 / (quaternions * quaternions).sum(-1)
#         o = torch.stack(
#             (
#                 1 - two_s * (j * j + k * k),
#                 two_s * (i * j - k * r),
#                 two_s * (i * k + j * r),
#                 two_s * (i * j + k * r),
#                 1 - two_s * (i * i + k * k),
#                 two_s * (j * k - i * r),
#                 two_s * (i * k - j * r),
#                 two_s * (j * k + i * r),
#                 1 - two_s * (i * i + j * j),
#             ),
#             -1,
#         )
#         return o.reshape(quaternions.shape[:-1] + (3, 3))
#
#     def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
#         if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#             raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
#
#         def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
#             ret = torch.zeros_like(x)
#             positive_mask = x > 0
#             ret[positive_mask] = torch.sqrt(x[positive_mask])
#             return ret
#
#         batch_dim = matrix.shape[:-2]
#         m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
#             matrix.reshape(batch_dim + (9,)), dim=-1
#         )
#
#         q_abs = _sqrt_positive_part(
#             torch.stack(
#                 [
#                     1.0 + m00 + m11 + m22,
#                     1.0 + m00 - m11 - m22,
#                     1.0 - m00 + m11 - m22,
#                     1.0 - m00 - m11 + m22,
#                 ],
#                 dim=-1,
#             )
#         )
#
#         quat_by_rijk = torch.stack(
#             [
#                 torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
#                 torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
#                 torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
#                 torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
#             ],
#             dim=-2,
#         )
#
#         flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
#         quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
#
#         return quat_candidates[
#                F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
#                ].reshape(batch_dim + (4,))
#
#     def R(self) -> torch.Tensor:
#         return self.quaternion_to_matrix(self.quaternion)
#
#     # @property
#     # def ref_R(self) -> torch.Tensor:
#     #     return self.quaternion_to_matrix(self.ref_quaternion)
#
#     # def __repr__(self):
#     #     print("222222222222222222")
#     #     print("self.uid: ", self.uid)
#     #     return f"[Camera {self.uid}] Quaternion: {self.quaternion.detach().squeeze().cpu().numpy().tolist()}, Translation: {self.T.detach().squeeze().cpu().numpy().tolist()}"
#
