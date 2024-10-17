from dataclasses import dataclass

import torch
from einops import reduce, rearrange
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim
#
# import cv2
import json
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
from ..dataset.types import Batch
from ..flow import Flows
from ..loss import Loss
from ..misc.image_io import prep_image
from ..tracking import Tracks
from ..visualization import Visualizer
from .model import Model, ModelExports
from ..export.colmap import export_to_colmap_e2e
from .render import render
from ..scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from ..arguments import ModelParams, PipelineParams, OptimizationParams
import sys
import torchvision.utils as vutils
import random
from pathlib import Path
from random import randint
import os
from ..utils.image_utils import psnr
from time import time
from ..utils.loss_utils import l1_loss, ssim
from ..export.flowmap2gs import flowmap_2_gs, save_json, xyz_from_flowmap, matrix_to_quaternion
from ..scene.dataset_readers import readFlowmapSceneInfo, readColmapSceneInfo, readSceneInfo
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, camera_from_camInfos_selection, pose_to_extrinsics
from torch.optim.lr_scheduler import StepLR
from ..misc.cropping import center_crop_intrinsics
from ..lpipsPyTorch import lpips


@dataclass
class ModelWrapperOverfitCfg:
    lr: float
    patch_size: int


class ModelWrapperOverfit(LightningModule):
    def __init__(
            self,
            cfg: ModelWrapperOverfitCfg,
            model: Model,  # flowmap model
            batch: Batch,  # crop后的batch，包括videos中每帧的具体数据, scenes, extrinsics、intrinsics
            flows: Flows,
            tracks: list[Tracks] | None,
            losses: list[Loss],
            visualizers: list[Visualizer],

            gaussians: GaussianModel,  # 3dgs的高斯模型
            background,  # 背景颜色
            frame_paths: list[Path],  # 保存的colmap文件路径
            pre_crop: tuple[int, int],  # crop前的图片大小
            batch_uncropped_videos,
            # : Float[Tensor, "batch frame 3 uncropped_height uncropped_width"],       # crop后的batch 的 color
            colmap_path: Path,  # 保存的colmap文件路径
            dataset_root: Path,  # 数据集的根目录
            opt_prams,
            args

    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch = batch
        self.flows = flows
        self.tracks = tracks
        self.model = model
        self.losses = losses
        self.visualizers = visualizers

        self.frame_paths = frame_paths
        self.uncropped_exports_shape = pre_crop  # (180,240)
        self.uncropped_videos = batch_uncropped_videos  # 3024  4032
        self.path = colmap_path
        self.pipeline = {"compute_cov3D_python": False, "convert_SHs_python": False, "debug": False}
        self.background = background

        self.gaussians = gaussians
        self.dataset_root = dataset_root
        self.opt_prams = opt_prams
        self.viewp = None
        self.cameras_extent = None
        self.sceneinfo = None
        self.depth = None
        self.intrinsics = None
        self.extrinsics = None
        self.train_cams = None
        self.args = args
        self.pose = None
        self.viewpoint_stack = None
        self.extrinsics_gs = None
        self.intrinsics_flowmap_2000 = None
    def to(self, device: torch.device) -> None:
        self.batch = self.batch.to(device)
        self.flows = self.flows.to(device)
        if self.tracks is not None:
            self.tracks = [tracks.to(device) for tracks in self.tracks]
        super().to(device)

    def training_step(self, dummy):
        choice = 1          # 选择模式, 1:flowmap      2:colmap
        iteration_gs = 0    # 决定多少iteration后进行gs render
        # xyz_flowmap = None
        num_images = self.args.num_images
        extrinsics_gs = False
        intrinsics_fixed, extrinsics_fixed = False, False      # True, True
        xyz_flowmap = True
        ### Step1. Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)  # model_output.surfaces:[1,3,H,W,3]   # self.global_step: 0
        point_3d = rearrange(model_output.surfaces, "b c h w d -> (b c h w) d")
        # print("self.batch.videos: ", self.batch.videos.shape)     # torch.Size([1, 20, 3, 160, 224])
        depths = model_output.depths  # ([1, 20, 160, 224])       # 128 256
        # depths_upsampled = torch.nn.functional.interpolate(depths, size=(1200, 1600), mode='bilinear', align_corners=False)
        extrinsics = model_output.extrinsics  # torch.Size([1, 20, 4, 4])  float32  tensor
        intrinsics = model_output.intrinsics  # torch.Size([1, 20, 3, 3])  float32  # 20个图像内参是共享的，一样的
        # if(self.global_step== 1999):
        #     print("instrinsic: ", intrinsics)
        #     with open("/data2/hkk/3dgs/flowmap/outputs/local/output/intrinsics.txt", "a") as file:
        #         file.write(f"[ITER {(self.global_step)}] input.shape: {self.batch.videos.shape}\n instrinsic : {intrinsics}\n")

        ### Step2. 计算render的相机参数和gaussian参数
        # 选择1： 直接处理flowmap输出的相机参数，并进行传递render函数；
        ### Step2.1 Compute viewpoint_cam       # 内参应该现有(160,224) → (180,240)
        if self.global_step >= iteration_gs:      # 从第2000次迭代进行3dgs render
            if self.global_step == iteration_gs:      # 从第2000次迭代进行3dgs render
                None
                # self.model.backbone.midas.pretrained.requires_grad_(False)
                # self.model.backbone.midas.scratch.requires_grad_(False)
                # self.model.backbone.midas_out.requires_grad_(False)

            _, _, h_cropped, w_cropped = depths.shape
            h_uncropped, w_uncropped = self.uncropped_exports_shape
            intrinsics_uncropped = center_crop_intrinsics(
                intrinsics,
                (h_cropped, w_cropped),  # 160  224
                (h_uncropped, w_uncropped),  # (180, 240)
            )
            if intrinsics_fixed:
                self.intrinsics = intrinsics_uncropped
            if extrinsics_fixed:
                self.extrinsics = model_output.extrinsics
            extrinsics_colmap, intrinsics_colmap = flowmap_2_gs(intrinsics_uncropped[0], extrinsics[0], self.frame_paths, self.uncropped_videos)  # 传入的是flowmap的输出，输出的是内外参   # self.uncropped_videos: [1,20,3,3024,4032]
            # scene_info = readFlowmapSceneInfo(extrinsics_colmap, intrinsics_colmap, self.dataset_root, eval=True)
            scene_info = readSceneInfo(extrinsics_colmap, intrinsics_colmap, self.dataset_root, eval=self.args.eval, num_images=num_images)

            ### 保存 cameras.json文件
            save_json(scene_info, self.opt_prams.model_path)

            # view_point_selcetion = randint(0, len(scene_info.train_cameras) - 1)  # 随机选一个视角
            # viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, scene_info.train_cameras, resolution_scale=1, args=self.args)
            # pose = self.gaussians.init_RT_seq(viewpoint_cam)
            # random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            self.scene_info = scene_info

            self.intrinsics_flowmap_2000 = model_output.intrinsics

            if extrinsics_gs:
                if not self.viewpoint_stack:
                    self.viewpoint_stack = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale=1, args=self.args)
                self.viewp = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
                self.pose = self.gaussians.get_RT(self.viewp.uid)
            elif intrinsics_fixed and extrinsics_fixed:
                view_point_selcetion = randint(0, len(self.scene_info.train_cameras) - 1)  # 随机选一个视角
                viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, self.scene_info.train_cameras, resolution_scale=1, args=self.args)
                # viewpoint_cam.cam_requires_grad_(False)    #
                self.viewp = viewpoint_cam
                self.pose = self.gaussians.init_RT_seq(self.viewp)    # 获得  torch.cat([quat, tran])
            else:
                view_point_selcetion = randint(0, len(scene_info.train_cameras) - 1)  # 随机选一个视角
                viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, scene_info.train_cameras, resolution_scale=1, args=self.args)
                # viewpoint_cam.cam_requires_grad_(False)    #
                pose = self.gaussians.init_RT_seq(viewpoint_cam)

        ### Step2.2 Initialization for gaussians
        if self.global_step > iteration_gs:  # 因为self.global_step从0开始
            self.gaussians.update_learning_rate((self.global_step - iteration_gs))  # 根据当前迭代次数更新学习率
            # lr = self.gaussians.update_learning_rate(self.global_step)
            # self.log("train/loss/learning_rate", lr)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.global_step % 1000 == 0 and self.global_step > iteration_gs:  # 每1000次迭代，提升球谐函数的次数以改进模型复杂度     # 初始为0阶
            self.gaussians.oneupSHdegree()

        # 选择1 点云通过unproject生成，并进一步生成render的gaussian参数
        if self.global_step >= iteration_gs:      # 2000
            if self.global_step == iteration_gs:  # 2000
                # self.depth = depths
                # 根据训练视角的索引，选出训练视角，进行点云初始化
                # self.gaussians.create_gaussian_params(intrinsics_uncropped, extrinsics, depths, self.batch, scene_info.nerf_normalization["radius"], num_images=num_images)
                self.gaussians.create_gaussian_params(intrinsics_uncropped, extrinsics, depths, self.batch, self.scene_info.nerf_normalization["radius"], self.opt_prams.model_path, num_images=num_images)
                # self.gaussians.create_gaussian_params(intrinsics_uncropped, extrinsics, depths, self.batch, self.scene_info.nerf_normalization["radius"], num_images=num_images)
                # self.gaussians.create_fewshot_gaussian_params(intrinsics_uncropped, extrinsics, depths, self.batch, scene_info.nerf_normalization["radius"], num_images=num_images)
                # viewpoint_stack = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1, args=None)
                # input = self.uncropped_videos.permute(0, 2, 1, 3, 4)    # [1, 20, 3, 3024, 4032]
                # # 然后，我们可以使用 interpolate 函数进行上采样
                # output = torch.nn.functional.interpolate(input, size=(20, 1200, 1600), mode='trilinear', align_corners=False)
                #
                # # 最后，我们需要将输出的尺寸调整回 [batch_size, depth, channels, height, width]
                # image_1200_1600 = output.permute(0, 2, 1, 3, 4)
                # self.gaussians.create_gaussian_params(intrinsics_uncropped, extrinsics, depths_upsampled, image_1200_1600, scene_info.nerf_normalization["radius"])  # self.batch: rgb
                # self.gaussians.get_xyz_from_depthflowmap(depths, intrinsics_uncropped, extrinsics, self.batch, view_point_selcetion, num_images=num_images)
                self.gaussians.training_setup(self.opt_prams)
            # 直接传递self.background: [0,0,0]

            # xyz_flowmap = xyz_from_flowmap(depths, intrinsics_uncropped, extrinsics, self.batch)    # num_images 默认-1
            # 将内参固定，外参固定，depth优化，反投影xyz；    第二部：外参变化
            # xyz_flowmap = xyz_from_flowmap(depths, intrinsics_uncropped, extrinsics, self.batch, num_images=num_images)   # few-shot when input all images
            # if intrinsics_fixed and extrinsics_fixed:
            #     xyz_flowmap = xyz_from_flowmap(depths, self.intrinsics, self.extrinsics, self.batch, num_images=num_images)               # 内参，cameras不变
            # else:
            # depth_mixed = 0.5 * self.gaussians.get_depth_from_flowmap + 0.5 * depths
            # depth_mixed = depths   # [1.0807, 1.0361, 0.9340
            # depth_mixed = self.gaussians.get_depth_from_flowmap   # [1.0807, 1.0361, 0.9340
            # depth_mixed = ((self.gaussians.get_depth_from_flowmap + depths) * self.gaussians.get_depth_scale) + self.gaussians.get_depth_shift
            depth_mixed = ((self.gaussians.get_depth_from_flowmap + depths))
            print("self.gaussians.get_depth_from_flowmap:", self.gaussians.get_depth_from_flowmap)
            print("self.gaussians.get_depth_scale:", self.gaussians.get_depth_scale)
            print("self.gaussians.get_depth_shift:", self.gaussians.get_depth_shift)
            self.gaussians.get_xyz_from_depthflowmap(depth_mixed, intrinsics_uncropped, extrinsics, self.batch, num_images=num_images)
                # self.gaussians.get_xyz_from_depthflowmap(depths, intrinsics_uncropped, extrinsics, self.batch, view_point_selcetion, num_images=num_images)
                # xyz_flowmap = xyz_from_flowmap(depths, intrinsics_uncropped, extrinsics, self.batch, num_images=num_images)
            # xyz_flowmap = xyz_from_flowmap(self.depth, intrinsics_uncropped, extrinsics, self.batch)        # depth不变

            ### Step2.3 3dgs render         # render函数要引入每次由depth和cameras生成的  xyz
            # render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline, self.background)
            if intrinsics_fixed and extrinsics_fixed:
                render_pkg = render(self.viewp, self.gaussians, self.pipeline, self.background, xyz_flowmap=xyz_flowmap, camera_pose=self.pose)
            else:
                # render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline, self.background, depths, intrinsics_uncropped, extrinsics, self.batch, xyz_flowmap=xyz_flowmap, camera_pose=pose)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline, self.background, xyz_flowmap=xyz_flowmap, camera_pose=pose)
            # render_pkg = render(self.viewp, self.gaussians, self.pipeline, self.background, camera_pose=self.pose)
            # render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline, self.background, xyz_flowmap)      # xyz取决于flowmap
            # render_pkg = render(self.viewp, self.gaussians, xyz_flowmap, self.pipeline, self.background)         # cameras不变  # 走这条路，用gs去优化RT
            render_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[   # torch.Size([3, 1200, 1600])   torch.Size([560, 3])  torch.Size([560])  torch.Size([560])
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if intrinsics_fixed and extrinsics_fixed:
                gt_image = self.viewp.original_image.cuda()
            else:
                gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(render_image, gt_image)  # 计算L1 loss
            loss_gs = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(render_image, gt_image))
            # Record L1 loss value
            self.log("train/loss/loss_gs", loss_gs)
            # print("loss_gs: ", loss_gs)

        # flowmap 计算corresponding loss
        total_loss = 0
        for loss_fn in self.losses:
            # print("loss_fn: ", loss_fn)         # LossFlow((mapping): MappingHuber())       LossTracking((mapping): MappingHuber())
            loss = loss_fn.forward(self.batch, self.flows, self.tracks, model_output, self.global_step)
            self.log(f"train/loss/{loss_fn.cfg.name}", loss)
            # print("loss: ", loss)  # tensor(7.2532, device='cuda:0', grad_fn=<MulBackward0>)        loss:  tensor(6.6514, device='cuda:0', grad_fn=<MulBackward0>)
            total_loss = total_loss + loss
        # print("total_loss: ", total_loss)
        # corresponding loss 加 rgb loss
        if self.global_step >= iteration_gs:
            # k = self.gaussians.xyz_scheduler_args(self.global_step)
            # k = self.global_step / self.opt_prams.max_steps
            # total_loss = (1-k) * total_loss + k * loss_gs
            total_loss = total_loss + loss_gs
            # loss_gs.backward()
        total_loss.backward()
        # if self.global_step == 32000:   # 32000
        #     self.extrinsics_gs = (pose_to_extrinsics(self.gaussians.P))    # [batch, N, 4, 4]
        #     # self.extrinsics_gs = (pose_to_extrinsics(self.gaussians.P)).unsqueeze(0)   # [batch, N, 4, 4]
        #     # extrinsics_gs = extrinsics_gs.unsequeeze(0)
        #     print("self.gaussians.P:", self.gaussians.P)
        #     print("extrinsics_gs:", self.extrinsics_gs)
        #
        #     quater = matrix_to_quaternion(self.extrinsics_gs)
        #     print("quater:", quater)
        #     self.extrinsics_gs = self.extrinsics_gs.unsqueeze(0)

        if self.global_step >= iteration_gs:
            # if self.global_step == iteration_gs:
            #     self.gaussians.optimizer.add_param_group({'params': self.gaussians.get_depth_from_flowmap, 'lr': self.gaussians.xyz_scheduler_args(self.global_step), 'name': 'depth'})
            #     self.gaussians.optimizer.add_param_group({'params': self.gaussians.get_xyz_from_flowmap, 'lr': self.gaussians.depth_scheduler_args(self.global_step), 'name': 'xyz_from_flowmap'})
            # if self.global_step == iteration_gs+1 :
            #     self.gaussians.optimizer.add_param_group({'params': self.gaussians.P, 'lr': self.gaussians.cam_scheduler_args(self.global_step), 'name': 'pose'})
            a_1 = self.gaussians.get_xyz_from_flowmap
            a_2 = self.gaussians.get_depth_from_flowmap
            a_3 = self.gaussians.get_features
            a_4 = self.gaussians.P
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            b_2 = self.gaussians.get_depth_from_flowmap
            b_3 = self.gaussians.get_features
            if torch.equal(a_2, b_2):
                print("get_depth_from_flowmap unchange")
            if torch.equal(a_3, b_3):
                print("get_depth_from_flowmap unchange")

        self.optimizers().step()
        self.optimizers().zero_grad()

        # Log intrinsics error.
        if self.batch.intrinsics is not None:
            fx_hat = reduce(model_output.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_hat = reduce(model_output.intrinsics[..., 1, 1], "b f ->", "mean")
            fx_gt = reduce(self.batch.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_gt = reduce(self.batch.intrinsics[..., 1, 1], "b f ->", "mean")
            self.log("train/intrinsics/fx_error", (fx_gt - fx_hat).abs())
            self.log("train/intrinsics/fy_error", (fy_gt - fy_hat).abs())



        if (self.global_step > iteration_gs):  # >
            with torch.no_grad():
                if (self.global_step % 5 == 0):   # 200
                # if (self.global_step % 5000 == 0) or (self.global_step == 3000) or (self.global_step == 11000) or (self.global_step == 12000) or (self.global_step == 13000) or (self.global_step == 14000) or (self.global_step == 32000):
                    print("self.global_step: ", self.global_step)
                    point_cloud_path = os.path.join(self.opt_prams.model_path, "point_cloud/iteration_{}".format(self.global_step))
                    # print("\n[ITER {}] Saving Checkpoint".format(self.global_step))
                    # torch.save((self.gaussians.capture(), self.global_step), "/data2/hkk/3dgs/flowmap/outputs/local/output" + "/chkpnt" + str(self.global_step) + ".pth")
                    if xyz_flowmap == None:
                        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                    else:
                        self.gaussians.save_ply_xyz(os.path.join(point_cloud_path, "point_cloud.ply"), depths, intrinsics_uncropped, extrinsics, self.batch, xyz_flowmap)
                        # self.gaussians.save_ply_xyz(os.path.join(point_cloud_path, "point_cloud.ply"), xyz_flowmap)
                    # camera_path = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output", "camera/iteration_{}".format(self.global_step))
                    # os.makedirs(camera_path, exist_ok=True)
                    if intrinsics_fixed and extrinsics_fixed:
                        viewpoint_stack = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale=1, args=self.args)
                        viewpoint_stack_test = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale=1, args=self.args)
                    else:
                        viewpoint_stack = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1, args=self.args)
                        viewpoint_stack_test = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale=1, args=self.args)

                    self.train_cams = {}
                    # self.train_cams[1] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale=1, args=self.args)
                    # # print("11111111: ", self.train_cams[1].state_dict())
                    # torch.save({scale: {cam.image_name: cam.state_dict() for cam in self.train_cams[scale]} for scale in
                    #             self.train_cams}, os.path.join(camera_path, "train_cameras.pkl"))

                    validation_configs = ({'name': 'test', 'cameras': viewpoint_stack_test},
                                          {'name': 'train', 'cameras': viewpoint_stack})  # 5, 10, 15, 20, 25
                                          # {'name': 'train', 'cameras': [viewpoint_stack[idx % len(viewpoint_stack)] for idx in range(5, 30, 5)]})  # 5, 10, 15, 20, 25
                    # 计算pnsr，并保存输出的render image
                    for config in validation_configs:
                        if config['cameras'] and len(config['cameras']) > 0:  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39   len(config['cameras'])==2
                            l1_test = 0.0
                            psnr_test = 0.0
                            ssim_test = 0.0
                            lpips_test = 0.0
                            for idx, viewpoint in enumerate(config['cameras']):  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39
                                if config['name'] == "train":
                                    # pose = self.gaussians.get_RT(viewpoint.uid)
                                    pose = self.gaussians.init_RT_seq(viewpoint)
                                # Create a directory to save the rendered images
                                rendered_dir = os.path.join(self.opt_prams.model_path, config['name'], "rendered_images")
                                os.makedirs(rendered_dir, exist_ok=True)

                                # Create a directory to save the ground truth images
                                gt_dir = os.path.join(self.opt_prams.model_path, config['name'], "gt_images")
                                os.makedirs(gt_dir, exist_ok=True)

                                # Loop through the cameras and render the images
                                # image = render(viewpoint, self.gaussians, self.pipeline, self.background)["render"]  # viewpoint: uid=0
                                # image = render(viewpoint, self.gaussians, self.pipeline, self.background, xyz_flowmap)["render"]  # viewpoint: uid=0
                                # image = render(viewpoint, self.gaussians, self.pipeline, self.background, camera_pose=pose)["render"]  # viewpoint: uid=0
                                # image = render(viewpoint_cam, self.gaussians, self.pipeline, self.background, depths, intrinsics_uncropped, extrinsics, self.batch, xyz_flowmap=xyz_flowmap, camera_pose=pose)["render"]  # viewpoint: uid=0
                                image = render(viewpoint, self.gaussians, self.pipeline, self.background, xyz_flowmap=xyz_flowmap, camera_pose=pose)["render"]  # viewpoint: uid=0
                                gt_image = viewpoint.original_image.to("cuda")

                                # Save the rendered image
                                # vutils.save_image(image, os.path.join(rendered_dir, f"{idx}.png"), normalize=True)
                                vutils.save_image(image, os.path.join(rendered_dir, f"{idx}.png"))

                                # Save the ground truth image
                                # vutils.save_image(gt_image, os.path.join(gt_dir, f"{idx}.png"), normalize=True)
                                vutils.save_image(gt_image, os.path.join(gt_dir, f"{idx}.png"))

                                image__ = torch.clamp(image, 0.0, 1.0)  # 将input的值限制在[min, max]之间
                                gt_image__ = torch.clamp(gt_image, 0.0, 1.0)
                                l1_test += torch.abs((image__ - gt_image__)).mean().double()
                                # render_11 = Image.open("/data2/hkk/3dgs/flowmap/outputs/local/output/test/rendered_images/0.png")
                                # gt_11 = Image.open("/data2/hkk/3dgs/flowmap/outputs/local/output/test/gt_images/0.png")
                                # render_1122 = tf.to_tensor(render_11).unsqueeze(0)[:, :3, :, :].cuda()
                                # gt_1122 = tf.to_tensor(gt_11).unsqueeze(0)[:, :3, :, :].cuda()
                                # psnr_test_11 = psnr(render_1122, gt_1122).mean().double()
                                # print("psnr_test_11: ", psnr_test_11)
                                psnr_test += psnr(image__, gt_image__).mean().double()    # 12.9664
                                # ssim_test += ssim(image__, gt_image__).mean().double()
                                # lpips_test += lpips(image__, gt_image__, net_type='vgg').mean().double()
                            psnr_test /= len(config['cameras'])
                            ssim_test /= len(config['cameras'])
                            lpips_test /= len(config['cameras'])
                            l1_test /= len(config['cameras'])
                            pnsr_test_path = os.path.join(self.opt_prams.model_path, "psnr.txt")
                            with open(pnsr_test_path, "a") as file:
                                file.write(f"[ITER {(self.global_step-2000)}] Evaluating {config['name']}: PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}\n")
                            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format((self.global_step-2000), config['name'], l1_test, psnr_test))

        # Optimizer step     # 执行优化器的一步，并准备下一次迭代


        # return total_loss

    ####  flowmap的validation，目前还未用到
    def validation_step(self, dummy):
        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)
        # print("enter into validation_step")

        # Generate visualizations.
        for visualizer in self.visualizers:
            visualizations = visualizer.visualize(
                self.batch,     # video [160,256]
                self.flows,
                self.tracks,
                model_output,
                self.model,
                self.global_step,
            )
            for key, visualization_or_metric in visualizations.items():
                if visualization_or_metric.ndim == 0:
                    # If it has 0 dimensions, it's a metric.
                    self.logger.log_metrics(
                        {key: visualization_or_metric},
                        step=self.global_step,
                    )
                else:
                    # If it has 3 dimensions, it's an image.
                    self.logger.log_image(
                        key,
                        [prep_image(visualization_or_metric)],
                        step=self.global_step,
                    )

    ###  定义model优化器
    def configure_optimizers(self) -> OptimizerLRScheduler:
        # for name, param in self.named_parameters():
        #     # with open('/data2/hkk/3dgs/flowmap/flowmap/model/names.txt', 'a') as f:
        #     #     f.write(name + '\n')
        #     print(name)
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    # flowmap 原始代码为输出colmap文件，未用到
    def export(self, device: torch.device) -> ModelExports:
        return self.model.export(
            self.batch.to(device),
            self.flows.to(device),
            self.global_step,
        )

    def export_gs(self, device: torch.device):
        return self.extrinsics_gs.to(device)


    def export_flowmap_2000_intrinsics(self, device: torch.device):
        return self.intrinsics_flowmap_2000.to(device)