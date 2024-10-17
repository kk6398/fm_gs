from dataclasses import dataclass

import torch
from einops import reduce
import tqdm
import torchvision
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
from ..export.flowmap2gs import flowmap_2_gs, save_json, xyz_from_flowmap
from ..scene.dataset_readers import readFlowmapSceneInfo, readColmapSceneInfo, readSceneInfo
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, camera_from_camInfos_selection
from ..utils.system_utils import searchForMaxIteration
from torch.optim.lr_scheduler import StepLR
from ..misc.cropping import center_crop_intrinsics
from ..lpipsPyTorch import lpips


@dataclass
class ModelWrapperOverfitevalCfg:
    lr: float
    patch_size: int


class ModelWrapperOverfit_eval(LightningModule):
    def __init__(
            self,
            cfg: ModelWrapperOverfitevalCfg,
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
        self.loaded_iter = None

    def to(self, device: torch.device) -> None:
        self.batch = self.batch.to(device)
        self.flows = self.flows.to(device)
        if self.tracks is not None:
            self.tracks = [tracks.to(device) for tracks in self.tracks]
        super().to(device)

    def training_step(self, dummy):
        iteration_gs = 3000    # 决定多少iteration后进行gs render
        num_images = self.args.num_images
        ### Step1. Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)  # self.global_step: 0
        # print("self.batch.videos: ", self.batch.videos.shape)     # torch.Size([1, 20, 3, 160, 224])
        depths = model_output.depths  # ([1, 20, 160, 224])       # 128 256
        # depths_upsampled = torch.nn.functional.interpolate(depths, size=(1200, 1600), mode='bilinear', align_corners=False)
        extrinsics = model_output.extrinsics  # torch.Size([1, 20, 4, 4])  float32  tensor
        intrinsics = model_output.intrinsics  # torch.Size([1, 20, 3, 3])  float32  # 20个图像内参是共享的，一样的
        if(self.global_step== 1999):

            print("instrinsic: ", intrinsics)
            with open("/data2/hkk/3dgs/flowmap/outputs/local/output/intrinsics.txt", "a") as file:
                file.write(f"[ITER {(self.global_step)}] input.shape: {self.batch.videos.shape}\n instrinsic : {intrinsics}\n")

        # flowmap 计算corresponding loss
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(
                self.batch, self.flows, self.tracks, model_output, self.global_step
            )
            self.log(f"train/loss/{loss_fn.cfg.name}", loss)
            # print(f"train/loss/{loss_fn.cfg.name}: {loss}")
            total_loss = total_loss + loss
        print("total_loss=====: ", total_loss)

        total_loss.backward()

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



        if self.global_step >= iteration_gs:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.args.model_path, "point_cloud"))
            print("Loading trained model at iteration {}".format(self.loaded_iter))
            self.gaussians.load_ply(os.path.join(self.args.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
            _, _, h_cropped, w_cropped = depths.shape
            h_uncropped, w_uncropped = self.uncropped_exports_shape
            intrinsics_uncropped = center_crop_intrinsics(
                intrinsics,
                (h_cropped, w_cropped),  # 160  224
                (h_uncropped, w_uncropped),  # (180, 240)
            )
            extrinsics_colmap, intrinsics_colmap = flowmap_2_gs(intrinsics_uncropped[0], extrinsics[0], self.frame_paths, self.uncropped_videos)
            all_images_path = os.path.join(self.dataset_root, "images")
            scene_info = readSceneInfo(extrinsics_colmap, intrinsics_colmap, all_images_path, eval=self.args.eval,
                                       num_images=num_images)

            viewpoint_stack_train = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1, args=self.args)
            viewpoint_stack_test = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale=1, args=self.args)

            validation_configs = ({'name': 'test', 'cameras': viewpoint_stack_test},
                                  {'name': 'train', 'cameras': viewpoint_stack_train}) # 5, 10, 15, 20, 25

            # rendered_dir = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output/test", "rendered_images")
            # os.makedirs(rendered_dir, exist_ok=True)
            #
            # # Create a directory to save the ground truth images
            # gt_dir = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output/test", "gt_images")
            # os.makedirs(gt_dir, exist_ok=True)


            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39   len(config['cameras'])==2
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39
                        rendered_dir = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output", config['name'],
                                                    "rendered_images")
                        os.makedirs(rendered_dir, exist_ok=True)

                        # Create a directory to save the ground truth images
                        gt_dir = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output", config['name'],
                                              "gt_images")
                        os.makedirs(gt_dir, exist_ok=True)

                        # Loop through the cameras and render the images
                        # image = render(viewpoint, self.gaussians, self.pipeline, self.background)["render"]  # viewpoint: uid=0
                        image = render(viewpoint, self.gaussians, self.pipeline, self.background)[
                            "render"]  # viewpoint: uid=0
                        gt_image = viewpoint.original_image.to("cuda")

                        # Save the rendered image
                        # vutils.save_image(image, os.path.join(rendered_dir, f"{idx}.png"), normalize=True)
                        vutils.save_image(image, os.path.join(rendered_dir, f"{idx}.png"))

                        # Save the ground truth image
                        # vutils.save_image(gt_image, os.path.join(gt_dir, f"{idx}.png"), normalize=True)
                        vutils.save_image(gt_image, os.path.join(gt_dir, f"{idx}.png"))

                        image__ = torch.clamp(image, 0.0, 1.0)  # 将input的值限制在[min, max]之间
                        gt_image__ = torch.clamp(gt_image, 0.0, 1.0)

                        # render_11 = Image.open("/data2/hkk/3dgs/flowmap/outputs/local/output/test/rendered_images/0.png")
                        # gt_11 = Image.open("/data2/hkk/3dgs/flowmap/outputs/local/output/test/gt_images/0.png")
                        # render_1122 = tf.to_tensor(render_11).unsqueeze(0)[:, :3, :, :].cuda()
                        # gt_1122 = tf.to_tensor(gt_11).unsqueeze(0)[:, :3, :, :].cuda()
                        # psnr_test_11 = psnr(render_1122, gt_1122).mean().double()
                        # print("psnr_test_11: ", psnr_test_11)
                        psnr_test += psnr(image__, gt_image__).mean().double()  # 12.9664
                        # ssim_test += ssim(image__, gt_image__).mean().double()
                        # lpips_test += lpips(image__, gt_image__, net_type='vgg').mean().double()
                    psnr_test /= len(config['cameras'])
                    print("\nPSNR: ", (psnr_test))


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
