import json
import shutil
from pathlib import Path
from time import time
import os
import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
from torch.utils.data import default_collate
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import sys
# Configure beartype and jaxtyping.


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with install_import_hook(
        ("flowmap",),
        ("beartype", "beartype"),
):
    from flowmap.config.common import get_typed_root_config
    from flowmap.config.overfit import OverfitCfg
    from flowmap.dataset import get_dataset
    from flowmap.dataset.data_module_overfit import DataModuleOverfit
    from flowmap.dataset.types import Batch
    from flowmap.export.colmap import export_to_colmap
    from flowmap.flow import compute_flows
    from flowmap.loss import get_losses
    from flowmap.misc.common_training_setup import run_common_training_setup
    from flowmap.misc.cropping import (
        crop_and_resize_batch_for_flow,
        crop_and_resize_batch_for_model,
    )
    from flowmap.model.model import Model
    from flowmap.model.model_wrapper_overfit import ModelWrapperOverfit
    from flowmap.tracking import compute_tracks
    from flowmap.visualization import get_visualizers

    # from .config.common import get_typed_root_config
    # from .config.overfit import OverfitCfg
    # from .dataset import get_dataset
    # from .dataset.data_module_overfit import DataModuleOverfit
    # from .dataset.types import Batch
    # from .export.colmap import export_to_colmap
    # from .flow import compute_flows
    # from .loss import get_losses
    # from .misc.common_training_setup import run_common_training_setup
    # from .misc.cropping import (
    #     crop_and_resize_batch_for_flow,
    #     crop_and_resize_batch_for_model,
    # )
    # from .model.model import Model
    # from .model.model_wrapper_overfit import ModelWrapperOverfit
    # from .tracking import compute_tracks
    # from .visualization import get_visualizers
from flowmap.scene import Scene, GaussianModel
from flowmap.arguments import ModelParams
# from .utils.loss_utils import l1_loss, ssim
#
# from .utils.image_utils import psnr
# # from .arguments import ModelParams, PipelineParams, OptimizationParams
# # import sys
# # from .utils.general_utils import safe_state
# from .scene.dataset_readers import readColmapSceneInfo
# from .utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, camera_from_camInfos_selection
# import random
# from random import randint
# from .model.render import render

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="overfit",
)
# def overfit(lp, op, pp, cfg_dict: DictConfig) -> None:
def overfit(cfg_dict: DictConfig) -> None:
    start_time = time()
    cfg = get_typed_root_config(cfg_dict, OverfitCfg)  # 获取cfg配置 参数
    # print("cfg : ", cfg)
    print("cfg.dataset.root: ", cfg.dataset[0].root)  # /data2/hkk/datasets/flowmap/llff_fern
    # wandb=WandbCfg(project='flowmap', mode='disabled', name='placeholder', group=None, tags=None)
    # checkpoint=CheckpointCfg(every_n_train_steps=2000, load='checkpoints/initialization_finetuned.ckpt'),
    # trainer=TrainerCfg(val_check_interval=50, max_steps=2000),
    # flow=FlowPredictorRaftCfg(name='raft', num_flow_updates=32, max_batch_size=8, show_progress_bar=True),
    # dataset=[DatasetImagesCfg(image_shape=None, scene=None, name='images', root=PosixPath('/data2/hkk/datasets/flowmap/llff_fern'))],
    # frame_sampler=FrameSamplerOverfitCfg(name='overfit', start=None, num_frames=None, step=None),
    # model=ModelCfg(backbone=BackboneMidasCfg(name='midas', pretrained=True, weight_sensitivity=None, mapping='original', model='MiDaS_small'),
    # intrinsics=IntrinsicsSoftminCfg(name='softmin', num_procrustes_points=8192, min_focal_length=0.5, max_focal_length=2.0, num_candidates=60, regression=RegressionCfg(after_step=1000, window=100)),
    # extrinsics=ExtrinsicsProcrustesCfg(name='procrustes', num_points=1000, randomize_points=False),
    # use_correspondence_weights=True),
    # loss=[LossFlowCfg(enable_after=0, weight=1000.0, name='flow', mapping=MappingHuberCfg(name='huber', delta=0.01)),
    #       LossTrackingCfg(enable_after=50, weight=100.0, name='tracking', mapping=MappingHuberCfg(name='huber', delta=0.01))],
    # visualizer=[VisualizerSummaryCfg(name='summary', num_vis_frames=8),
    # VisualizerTrajectoryCfg(name='trajectory', generate_plot=True, ate_save_path=None)],
    # cropping=CroppingCfg(image_shape=43200, flow_scale_multiplier=4, patch_size=32),
    # tracking=TrackPredictorCoTrackerCfg(name='cotracker', grid_size=35, similarity_threshold=0.2),
    # track_precomputation=TrackPrecomputationCfg(cache_path=None, interval=5, radius=20),
    # model_wrapper=ModelWrapperOverfitCfg(lr=3e-05, patch_size=32), local_save_root=None)

    # 初始化guassians模型
    gaussians = GaussianModel(cfg.trainer.sh_degree)       # 3
    # pipe = {"compute_cov3D_python": False, "convert_SHs_python": False, "debug": False}
    bg_color = [1, 1, 1] if cfg.trainer.white_background else [0, 0, 0]  # 设置背景颜色，根据数据集是否有白色背景来选择。     # white_background：False
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转化为 PyTorch Tensor，并移到 GPU 上
    # gaussians.training_setup(cfg.trainer)

    parser = ArgumentParser(description="cfg_args")
    lp = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    lp.extract(args)

    callbacks, logger, checkpoint_path, output_dir = run_common_training_setup(
        # 获取回调函数(监控学习率的变化)、日志记录器、checkpoints路径、输出目录
        cfg, cfg_dict
    )

    device = torch.device("cuda:0")
    colmap_path = output_dir / "colmap"



    # training_cfg = {"device": device, "eval": False, "images": 'images', "sh_degree": cfg.trainer.sh_degree,
    #                 "model_path": cfg.trainer.model_path, "white_background": cfg.trainer.white_background}
    os.makedirs(cfg.trainer.model_path, exist_ok=True)
    with open(os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/output", "cfg_args"), 'w') as cfg_log_f:
        # cfg_log_f.write(str(training_cfg))
        cfg_log_f.write(str(Namespace(**vars(lp))))

    # Load the full-resolution batch.
    dataset = get_dataset(cfg.dataset, "train", cfg.frame_sampler)  # 获取数据集  # dataset:  <flowmap.dataset.dataset_merged.DatasetMerged object at 0x7fb2be2e2710>
    batch = next(iter(dataset))  # 获取数据集的一个batch
    # 'frame_paths': 20张图像的路径 [PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG'),
    frame_paths = batch.pop("frame_paths", None)  # 获取batch中的frame_paths, 即每个视频帧的路径
    # print("frame_paths: ", frame_paths)                      # frame_paths:  [PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG'),
    if frame_paths is not None:
        frame_paths = [Path(path) for path in
                       frame_paths]  # 将frame_paths转换为Path对象    使用 Path 对象可以更方便地进行文件路径的操作，例如获取文件名、父目录、扩展名等。
    batch = Batch(**default_collate([batch]))  # 将batch中的数据进行拼接

    # Compute optical flow and tracks.                      # batch_for_model: 裁剪后的batch、图像内参、videos
    batch_for_model, pre_crop = crop_and_resize_batch_for_model(batch,      # batch_for_model.videos  torch.Size([1, 20, 3, 160, 224])    # pre_crop:  (180, 240)
                                                                cfg.cropping)  # 对batch进行裁剪和缩放  ###    pre_crop:裁剪后的H W

    batch_for_flow = crop_and_resize_batch_for_flow(batch, cfg.cropping)  # 对batch进行裁剪和缩放， 针对光流  #  batch_for_flow.videos  torch.Size([1, 20, 3, 640, 896])

    _, f, _, h, w = batch_for_model.videos.shape  # 获取batch中的视频的帧数、高、宽
    flows = compute_flows(batch_for_flow, (h, w), device, cfg.flow)  # 计算光流,  输入是裁剪之后的batch, 图像的高和宽， 设备， cfg.flow

    # Only compute tracks if the tracking loss is enabled.
    if any([loss.name == "tracking" for loss in cfg.loss]):  # 如果损失函数中有tracking
        tracks = compute_tracks(  # 计算tracks    # CoTracker V1
            batch_for_flow, device, cfg.tracking, cfg.track_precomputation
        )
    else:
        tracks = None

    # Set up the model.
    optimization_start_time = time()  #
    model = Model(cfg.model, num_frames=f, image_shape=(h, w))  # 模型初始化
    losses = get_losses(cfg.loss)  # loss初始化    # 对flow、tracking损失进行带有权重初始化的设置
    visualizers = get_visualizers(cfg.visualizer)  # 可视化
    model_wrapper = ModelWrapperOverfit(  # 模型包装器
        cfg.model_wrapper,  # ModelWrapperOverfitCfg(lr=3e-05, patch_size=32), local_save_root=None)
        model,  # 初始化的模型
        batch_for_model,  # crop后的batch，包括videos中每帧的具体数据, scenes, extrinsics、intrinsics
        flows,  # 计算出的光流
        tracks,  # 计算出的 tracks: CoTracker V1     facebookresearch_co-tracker_v1.0
        losses,  # 损失函数
        visualizers,

        gaussians,       # gaussian 模型
        background,      # 背景颜色
        frame_paths,     # 所有图像的路径
        pre_crop,        # 裁剪后的H W
        batch.videos,    # original rgb图像          batch.videos.shape   [1, 20, 3, 3024, 4032]
        colmap_path,     # colmap路径    未用到
        cfg.dataset[0].root,    # 读取图像的上级路径      # /data2/hkk/datasets/flowmap/llff_fern
        cfg.trainer             # 包含了optimizer的参数
    )  # 输出: 整个model模型

    # Only load the model's saved state (so that optimization restarts).
    if checkpoint_path is not None:  # 如果有checkpoint路径
        checkpoint = torch.load(checkpoint_path)
        model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer = Trainer(  # PyTorch Lightning 训练器初始化 #
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(  # 策略：是否使用ddp分布式训练
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,  # 50
        max_steps=cfg.trainer.max_steps,  # 2000，        30000     # cfg.trainer.max_steps
        log_every_n_steps=1,
    )
    trainer.fit(  # 训练
        model_wrapper,
        datamodule=DataModuleOverfit(),
    )

    # Export the result.
    print("Exporting results.")



    ### 后续的生成colmap文件，，未用到
    model_wrapper.to(device)
    exports = model_wrapper.export(device)
    # colmap_path = output_dir / "colmap"
    export_to_colmap(
        exports,
        frame_paths,
        pre_crop,
        batch.videos,
        colmap_path,
    )
    shutil.make_archive(colmap_path, "zip", output_dir, "colmap")  # 压缩文件

    # scene_info = readColmapSceneInfo(colmap_path, images="images", eval=False)
    # random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
    # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
    # # view_point_selcetion = randint(0, len(scene_info.train_cameras) - 1)  # 随机选一个视角
    # # viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, scene_info.train_cameras,
    # #                                                resolution_scale=1, args=None)
    # gaussians.create_from_pcd(scene_info.point_cloud, scene_info.nerf_normalization['radius'])
    # gaussians.training_setup(cfg.trainer)
    # first_iter = 0
    # progress_bar = tqdm(range(first_iter, cfg.trainer.max_steps), desc="Training progress")
    # first_iter += 1
    # ema_loss_for_log = 0.0
    # for iteration in range(first_iter, cfg.trainer.max_steps + 1):
    #     gaussians.update_learning_rate(iteration)
    #     if iteration % 1000 == 0:  # 每1000次迭代，提升球谐函数的次数以改进模型复杂度    初始的球谐函数次数为多少？？？
    #         gaussians.oneupSHdegree()
    #     view_point_selcetion = randint(0, len(scene_info.train_cameras) - 1)  # 随机选一个视角
    #     viewpoint_cam = camera_from_camInfos_selection(view_point_selcetion, scene_info.train_cameras,
    #                                                    resolution_scale=1, args=None)
    #     render_pkg = render(viewpoint_cam, gaussians, pipe, background)  # 渲染当前视角的图像
    #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
    #     render_pkg["visibility_filter"], render_pkg["radii"]
    #
    #     gt_image = viewpoint_cam.original_image.cuda()
    #     Ll1 = l1_loss(image, gt_image)  # tensor(0.1917, device='cuda:0', grad_fn=<MeanBackward0>)     10%进程时，下降至 0.01
    #     # print("Ll1: ", Ll1)
    #     loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image, gt_image))  # opt.lambda_dssim: 0.2  #计算渲染的图像与真实图像之间的loss
    #     loss.backward()  # 更新损失。loss反向传播
    #
    #     if iteration % 2000 == 0:  # 0
    #         point_cloud_path = os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/colmap/output",
    #                                         "point_cloud/iteration_{}".format(iteration))
    #         gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    #         viewpoint_stack = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1, args=None)
    #         validation_configs = ({'name': 'test', 'cameras': []},
    #                               {'name': 'train', 'cameras': [viewpoint_stack[idx % len(viewpoint_stack)] for idx in
    #                                                             range(5, 30, 5)]})  # 5, 10, 15, 20, 25
    #         # 计算pnsr，并保存输出的render image
    #         for config in validation_configs:
    #             if config['cameras'] and len(config['cameras']) > 0:  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39   len(config['cameras'])==2
    #                 l1_test = 0.0
    #                 psnr_test = 0.0
    #                 for idx, viewpoint in enumerate(
    #                         config['cameras']):  # config['cameras'] 一共5个  image_name: 128, 6, 59, 51, 39
    #                     import torchvision.utils as vutils
    #
    #                     # Create a directory to save the rendered images
    #                     rendered_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/rendered_images"
    #                     os.makedirs(rendered_dir, exist_ok=True)
    #
    #                     # Create a directory to save the ground truth images
    #                     gt_dir = "/data2/hkk/3dgs/flowmap/outputs/local/output/gt_images"
    #                     os.makedirs(gt_dir, exist_ok=True)
    #
    #                     # Loop through the cameras and render the images
    #                     image = render(viewpoint, gaussians, pipe, background)["render"]  # viewpoint: uid=0
    #                     gt_image = viewpoint.original_image.to("cuda")
    #
    #                     # Save the rendered image
    #                     vutils.save_image(image, os.path.join(rendered_dir, f"rendered_{idx}.png"), normalize=True)
    #
    #                     # Save the ground truth image
    #                     vutils.save_image(gt_image, os.path.join(gt_dir, f"gt_{idx}.png"), normalize=True)
    #
    #                     image__ = torch.clamp(image, 0.0, 1.0)  # 将input的值限制在[min, max]之间
    #                     gt_image__ = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
    #                     l1_test += torch.abs((image__ - gt_image__)).mean().double()
    #                     psnr_test += psnr(image__, gt_image__).mean().double()
    #                 psnr_test /= len(config['cameras'])
    #                 l1_test /= len(config['cameras'])
    #                 with open("psnr_test.txt", "a") as file:
    #                     file.write(f"[ITER {iteration}] Evaluating {config['name']}: PSNR {psnr_test}\n")
    #                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
    #
    #     with torch.no_grad():
    #         # Progress bar      # #记录损失的指数移动平均值，并定期更新进度条
    #         ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    #         if iteration % 10 == 0:
    #             progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
    #             progress_bar.update(10)
    #         if iteration == 30000:  # 30000
    #             progress_bar.close()
    #         # Densification   # 在指定迭代区间内，对3D高斯模型进行增密和修剪
    #         if iteration < cfg.trainer.densify_until_iter and iteration > 0:  # 在达到指定的迭代次数之前执行以下操作。   # opt.densify_until_iter: 15000
    #             # Keep track of max radii in image-space for pruning
    #             gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
    #                                                                       radii[
    #                                                                           visibility_filter])  # 将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考。
    #             gaussians.add_densification_stats(viewspace_point_tensor,
    #                                                    visibility_filter)  # 将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器。
    #             # 500                                                                 # 100
    #             if iteration > cfg.trainer.densify_from_iter and iteration % cfg.trainer.densification_interval == 0:  # 在指定的迭代次数之后，每隔一定的迭代间隔进行以下密集化操作。
    #                 size_threshold = 20 if iteration > cfg.trainer.opacity_reset_interval else None  # 根据当前迭代次数设置密集化的阈值。如果当前迭代次数大于 opt.opacity_reset_interval，则设置 size_threshold 为 20，否则为 None。
    #                 gaussians.densify_and_prune(cfg.trainer.densify_grad_threshold, 0.005,
    #                                                  scene_info.nerf_normalization['radius'],
    #                                                  size_threshold)  # 执行密集化和修剪操作，其中包括梯度阈值、密集化阈值、相机范围和之前计算的 size_threshold。
    #                 # 3000                                                                              # 500
    #             if iteration % cfg.trainer.opacity_reset_interval == 0 or (
    #                     cfg.trainer.white_background and iteration == cfg.trainer.densify_from_iter):  # 在每隔一定迭代次数或在白色背景数据集上的指定迭代次数时，执行以下操作。
    #                 gaussians.reset_opacity()  # 重置模型中的某些参数，涉及到透明度的操作，具体实现可以在 reset_opacity 方法中找到。
    #         gaussians.optimizer.step()  # 更新网络参数
    #         gaussians.optimizer.zero_grad(set_to_none=True)


    if cfg.local_save_root is not None:  # 是None     #  如果有本地保存路径
        # Save the COLMAP-style output.
        cfg.local_save_root.mkdir(exist_ok=True, parents=True)  # 创建本地保存路径
        shutil.copytree(colmap_path, cfg.local_save_root, dirs_exist_ok=True)  # 复制文件

        # Save the runtime. For a fair comparison with COLMAP, we record how long it
        # takes until the COLMAP-style output has been saved.
        times = {  # 记录时间
            "runtime": time() - start_time,
            "optimization_runtime": time() - optimization_start_time,
        }
        with (cfg.local_save_root / "runtime.json").open("w") as f:  # 保存时间
            json.dump(times, f)

            # Save the exports (poses, intrinsics, depth maps + corresponding colors).
        torch.save(exports, cfg.local_save_root / "exports.pt")  # 保存导出结果

        # Save a checkpoint.
        trainer.save_checkpoint(cfg.local_save_root / "final.ckpt")  # 保存checkpoint

    if cfg.wandb.mode != "disabled":  # 如果wandb不是禁用状态
        artifact = wandb.Artifact(f"colmap_{wandb.run.id}", type="colmap")  # 创建wandb artifact
        artifact.add_file(f"{colmap_path}.zip", name="colmap.zip")  # 添加文件
        wandb.log_artifact(artifact)  # 记录artifact
        artifact.wait()  # 等待artifact


if __name__ == "__main__":  # 主函数
    overfit()

