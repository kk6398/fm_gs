from dataclasses import dataclass

import torch
from einops import reduce
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim
#
# import cv2
import json
import numpy as np
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
from ..export.flowmap2gs import flowmap_2_gs, save_json, xyz_from_flowmap
from ..scene.dataset_readers import readFlowmapSceneInfo, readColmapSceneInfo
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, camera_from_camInfos_selection
from torch.optim.lr_scheduler import StepLR
from ..misc.cropping import center_crop_intrinsics


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
            opt_prams

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
    def to(self, device: torch.device) -> None:
        self.batch = self.batch.to(device)
        self.flows = self.flows.to(device)
        if self.tracks is not None:
            self.tracks = [tracks.to(device) for tracks in self.tracks]
        super().to(device)

    def training_step(self, dummy):
        choice = 1          # 选择模式, 1:flowmap      2:colmap
        iteration_gs = 2000
        ### Step1. Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)  # self.global_step: 0
        # print("self.batch.videos: ", self.batch.videos.shape)     # torch.Size([1, 20, 3, 160, 224])
        depths = model_output.depths  # ([1, 20, 160, 224])
        extrinsics = model_output.extrinsics  # torch.Size([1, 20, 4, 4])   tensor
        intrinsics = model_output.intrinsics  # torch.Size([1, 20, 3, 3])   # 20个图像内参是共享的，一样的

        ### Step2. 计算render的相机参数和gaussian参数
        # 选择1： 直接处理flowmap输出的相机参数，并进行传递render函数；
        ### Step2.1 Compute viewpoint_cam       # 内参应该现有(160,224) → (180,240)
        if self.global_step >= iteration_gs:      # 从第2000次迭代进行3dgs render
            if choice == 1:
                # if self.global_step == iteration_gs:
                _, _, h_cropped, w_cropped = depths.shape
                h_uncropped, w_uncropped = self.uncropped_exports_shape
                intrinsics_uncropped = center_crop_intrinsics(
                    intrinsics,
                    (h_cropped, w_cropped),  # 160  224
                    (h_uncropped, w_uncropped),  # (180, 240)
                )
                extrinsics_colmap, intrinsics_colmap = flowmap_2_gs(intrinsics_uncropped[0], extrinsics[0], self.frame_paths,
                                                                    self.uncropped_videos)  # 传入的是flowmap的输出，输出的是
                scene_info = readFlowmapSceneInfo(extrinsics_colmap, intrinsics_colmap, self.dataset_root)
                # self.sceneinfo = scene_info

                ### 保存 cameras.json文件
                save_json(scene_info)

                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

                view_point_selcetion = randint(0, len(scene_info.train_cameras) - 1)  # 随机选一个视角
                viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, scene_info.train_cameras,
                                                               resolution_scale=1, args=None)
                # self.viewp = viewpoint_cam
                # self.cameras_extent = self.sceneinfo.nerf_normalization["radius"]
            else:  # 选择2 通过生成colmap文件 cameras.bin images.bin，在读取colmap文件，生成对应相机参数
                # if self.global_step == iteration_gs:
                path_colmap = Path("/data2/hkk/3dgs/flowmap/outputs/local/colmap")
                export_to_colmap_e2e(self.batch.videos, depths, intrinsics, extrinsics, self.frame_paths,
                                     self.uncropped_exports_shape, self.uncropped_videos, path=path_colmap)
                scene_info = readColmapSceneInfo(path_colmap, images="images", eval=False)
                # self.sceneinfo = scene_info

                ### 保存 cameras.json文件
                save_json(scene_info)

                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
                view_point_selcetion = randint(0, len(scene_info.train_cameras) - 1)  # 随机选一个视角
                viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, scene_info.train_cameras,
                                                               resolution_scale=1, args=None)
                # self.viewp = viewpoint_cam
                # self.cameras_extent = self.sceneinfo.nerf_normalization["radius"]
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
                if choice == 1:
                    self.gaussians.create_gaussian_params(intrinsics_uncropped, extrinsics, depths, self.batch,
                                                          scene_info.nerf_normalization["radius"])  # self.batch: rgb
                else:  # 选择2 根据读取的colmap点云文件pcd，直接生成gaussian参数
                    self.gaussians.create_from_pcd(scene_info.point_cloud, self.scene_info.nerf_normalization['radius'])
                self.gaussians.training_setup(self.opt_prams)
            # 直接传递self.background: [0,0,0]

            xyz_flowmap = xyz_from_flowmap(depths, intrinsics_uncropped, extrinsics, self.batch)

            ### Step2.3 3dgs render         # render函数要引入每次由depth和cameras生成的  xyz
            # render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline, self.background)
            render_pkg = render(viewpoint_cam, self.gaussians, xyz_flowmap, self.pipeline, self.background)
            render_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[   # torch.Size([3, 1200, 1600])   torch.Size([560, 3])  torch.Size([560])  torch.Size([560])
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(render_image, gt_image)  # 计算L1 loss
            loss_gs = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(render_image, gt_image))
            # Record L1 loss value
            self.log("train/loss/loss_gs", loss_gs)
            print("loss_gs: ", loss_gs)

        # flowmap 计算corresponding loss
        total_loss = 0
        for loss_fn in self.losses:
            # print("loss_fn: ", loss_fn)         # LossFlow((mapping): MappingHuber())       LossTracking((mapping): MappingHuber())
            loss = loss_fn.forward(
                self.batch, self.flows, self.tracks, model_output, self.global_step
            )
            self.log(f"train/loss/{loss_fn.cfg.name}", loss)
            # print("loss: ", loss)  # tensor(7.2532, device='cuda:0', grad_fn=<MulBackward0>)        loss:  tensor(6.6514, device='cuda:0', grad_fn=<MulBackward0>)
            total_loss = total_loss + loss
        # corresponding loss 加 rgb loss
        if self.global_step >= iteration_gs:
            # total_loss = total_loss + loss_gs
            loss_gs.backward()
        total_loss.backward()

        if (self.global_step > iteration_gs) and (self.global_step <= 7000):     # 2000~7000次迭代，每200次迭代进行一次测试
            if (self.global_step % 200 == 0):   # 0
                print("self.global_step: ", self.global_step)
                point_cloud_path = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output",
                                                "point_cloud/iteration_{}".format(self.global_step))
                self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                viewpoint_stack = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1, args=None)
                validation_configs = ({'name': 'test', 'cameras': []},
                                      {'name': 'train', 'cameras': [viewpoint_stack[idx % len(viewpoint_stack)] for idx in
                                                                    range(5, 30, 5)]})  # 5, 10, 15, 20, 25
                # 计算pnsr，并保存输出的render image
                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39   len(config['cameras'])==2
                        l1_test = 0.0
                        psnr_test = 0.0
                        for idx, viewpoint in enumerate(config['cameras']):  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39
                             # Create a directory to save the rendered images
                            rendered_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/rendered_images"
                            os.makedirs(rendered_dir, exist_ok=True)

                            # Create a directory to save the ground truth images
                            gt_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/gt_images"
                            os.makedirs(gt_dir, exist_ok=True)

                            # Loop through the cameras and render the images
                            image = render(viewpoint_cam, self.gaussians, xyz_flowmap, self.pipeline, self.background)[
                                "render"]  # viewpoint: uid=0
                            gt_image = viewpoint.original_image.to("cuda")

                            # Save the rendered image
                            vutils.save_image(image, os.path.join(rendered_dir, f"rendered_{idx}.png"), normalize=True)

                            # Save the ground truth image
                            vutils.save_image(gt_image, os.path.join(gt_dir, f"gt_{idx}.png"), normalize=True)

                            image__ = torch.clamp(image, 0.0, 1.0)  # 将input的值限制在[min, max]之间
                            gt_image__ = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            l1_test += torch.abs((image__ - gt_image__)).mean().double()
                            psnr_test += psnr(image__, gt_image__).mean().double()
                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])
                        with open("psnr_test.txt", "a") as file:
                            file.write(f"[ITER {self.global_step}] Evaluating {config['name']}: PSNR {psnr_test}\n")
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(self.global_step, config['name'], l1_test,
                                                                                psnr_test))
        if self.global_step > 7000:               # 7000~30000w次迭代，每3000次迭代进行一次测试
            if (self.global_step % 3000 == 0):  # 0
                print("self.global_step: ", self.global_step)
                point_cloud_path = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output",
                                                "point_cloud/iteration_{}".format(self.global_step))
                self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                viewpoint_stack = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1,
                                                           args=None)
                validation_configs = ({'name': 'test', 'cameras': []},
                                      {'name': 'train',
                                       'cameras': [viewpoint_stack[idx % len(viewpoint_stack)] for idx in
                                                   range(5, 30, 5)]})  # 5, 10, 15, 20, 25
                # 计算pnsr，并保存输出的render image
                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39   len(config['cameras'])==2
                        l1_test = 0.0
                        psnr_test = 0.0
                        for idx, viewpoint in enumerate(
                                config['cameras']):  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39
                            # Create a directory to save the rendered images
                            rendered_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/rendered_images"
                            os.makedirs(rendered_dir, exist_ok=True)

                            # Create a directory to save the ground truth images
                            gt_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/gt_images"
                            os.makedirs(gt_dir, exist_ok=True)

                            # Loop through the cameras and render the images
                            image = render(viewpoint_cam, self.gaussians, xyz_flowmap, self.pipeline, self.background)[
                                "render"]  # viewpoint: uid=0
                            gt_image = viewpoint.original_image.to("cuda")

                            # Save the rendered image
                            vutils.save_image(image, os.path.join(rendered_dir, f"rendered_{idx}.png"),
                                              normalize=True)

                            # Save the ground truth image
                            vutils.save_image(gt_image, os.path.join(gt_dir, f"gt_{idx}.png"), normalize=True)

                            image__ = torch.clamp(image, 0.0, 1.0)  # 将input的值限制在[min, max]之间
                            gt_image__ = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            l1_test += torch.abs((image__ - gt_image__)).mean().double()
                            psnr_test += psnr(image__, gt_image__).mean().double()
                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])
                        with open("psnr_test.txt", "a") as file:
                            file.write(f"[ITER {self.global_step}] Evaluating {config['name']}: PSNR {psnr_test}\n")
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(self.global_step, config['name'],
                                                                                l1_test,
                                                                                psnr_test))
        ### densification
#         if self.global_step >= iteration_gs:
#             with torch.no_grad():
#                 # Densification   # 在指定迭代区间内，对3D高斯模型进行增密和修剪
#                 if self.global_step < self.opt_prams.densify_until_iter and self.global_step > 0:  # 在达到指定的迭代次数之前执行以下操作。   # opt.densify_until_iter: 15000
#                     # Keep track of max radii in image-space for pruning
#                     self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
#                                                                               radii[
#                                                                                   visibility_filter])  # 将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考。
#                     self.gaussians.add_densification_stats(viewspace_point_tensor,
#                                                            visibility_filter)  # 将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器。
#                     # 500                                                                 # 100
#                     if self.global_step > self.opt_prams.densify_from_iter and self.global_step % self.opt_prams.densification_interval == 0:  # 在指定的迭代次数之后，每隔一定的迭代间隔进行以下密集化操作。
#                         size_threshold = 20 if self.global_step > self.opt_prams.opacity_reset_interval else None  # 根据当前迭代次数设置密集化的阈值。如果当前迭代次数大于 opt.opacity_reset_interval，则设置 size_threshold 为 20，否则为 None。
# -                        self.gaussians.densify_and_prune(self.opt_prams.densify_grad_threshold, 0.005,
#                                                          scene_info.nerf_normalization["radius"],   # self.cameras_extent
#                                                          size_threshold)  # 执行密集化和修剪操作，其中包括梯度阈值、密集化阈值、相机范围和之前计算的 size_threshold。
#                         # 3000                                                                              # 500
#                     if self.global_step % self.opt_prams.opacity_reset_interval == 0 or (
#                             self.opt_prams.white_background and self.global_step == self.opt_prams.densify_from_iter):  # 在每隔一定迭代次数或在白色背景数据集上的指定迭代次数时，执行以下操作。
#                         self.gaussians.reset_opacity()  # 重置模型中的某些参数，涉及到透明度的操作，具体实现可以在 reset_opacity 方法中找到。


        # Optimizer step     # 执行优化器的一步，并准备下一次迭代
        if self.global_step >= iteration_gs:
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

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

        # return total_loss

    ####  flowmap的validation，目前还未用到
    def validation_step(self, dummy):
        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)
        # print("enter into validation_step")

        # Generate visualizations.
        for visualizer in self.visualizers:
            visualizations = visualizer.visualize(
                self.batch,
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
        #     with open('/data2/hkk/3dgs/flowmap/flowmap/model/names.txt', 'a') as f:
        #         f.write(name + '\n')
        #     print(name)
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    # flowmap 原始代码为输出colmap文件，未用到
    def export(self, device: torch.device) -> ModelExports:
        return self.model.export(
            self.batch.to(device),
            self.flows.to(device),
            self.global_step,
        )
