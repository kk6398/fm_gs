#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# #
# import os
# import json
# # from dnnlib import EasyDict
# import torch
# from torch.nn import functional as F
# # from utils.loss_utils import norm_xy, predict_hierarchical_correspondence
# # from gaussian_renderer import renderApprSurface

# import sys
# import numpy as np

# from tqdm import tqdm

# # from argparse import ArgumentParser
# # from .arguments import ModelParams, PipelineParams, OptimizationParams
# from matplotlib import pyplot as plt
# # from argparse import Namespace
# # from utils.eval_pose import compute_rpe, compute_ATE
# from .utils.image_utils import psnr
# from flowmap.scene import Scene, GaussianModel
# from .model.render import render



import json
import shutil
from pathlib import Path
from time import time
import os
import matplotlib.pyplot as plt
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
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
from flowmap.utils.image_utils import psnr
from flowmap.utils.pose_utils import get_tensor_from_camera
from flowmap.arguments import ModelParams, PipelineParams, OptimizationParams

# # import sys
# from .scene.dataset_readers import readColmapSceneInfo
# from .utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, camera_from_camInfos_selection
# import random
# from random import randint
from flowmap.model.render import render, render_eval
from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def read_cfg(path: str):
    assert os.path.exists(os.path.join(path, 'cfg_args'))
    with open(os.path.join(path, 'cfg_args')) as f:
        string = f.read()
    args = eval(string)
    return EasyDict(**vars(args).copy())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import numpy as np
from PIL import Image
from torchvision.utils import make_grid
@torch.no_grad()
def render_tensor(img: torch.Tensor, normalize: bool = False, nrow: int = 8) -> Image.Image:
    def process_dtype(img):
        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 255.
            if normalize:
                img = img * 2 - 1
        return img
    if type(img) == list:
        img = torch.cat([process_dtype(i) if len(i.shape) == 4 else process_dtype(i[None, ...]) for i in img], dim=0).expand(-1, 3, -1, -1)
    elif len(img.shape) == 3:
        img = process_dtype(img).expand(3, -1, -1)
    elif len(img.shape) == 4:
        img = process_dtype(img).expand(-1, 3, -1, -1)
    
    img = img.squeeze()
    
    if normalize:
        img = img / 2 + .5
    
    if len(img.shape) == 3:
        return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 2:
        return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 4:
        return Image.fromarray((make_grid(img, nrow=nrow).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def register(dataset, opt, pipe, load_iteration, eval_pose):
    load_iteration = dataset.load_iteration                # load_iteration = dataset.load_iteration
    # dataset = read_cfg("/data2/hkk/3dgs/flowmap/outputs/local_ttfamily12_200/output")      # dataset = read_cfg(dataset.model_path) 
    print("dataset: ", dataset)
    # dataset.load_iteration = load_iteration
    gaussians = GaussianModel(dataset.sh_degree)     # gaussians = GaussianModel(dataset.sh_degree)
    # checkpoint = "/data2/hkk/3dgs/flowmap/outputs/local/output/chkpnt4.pth"
    # if checkpoint:  # 如果提供了checkpoint，则从checkpoint加载模型参数并恢复训练进度
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)
    scene = Scene(dataset, gaussians, load_iteration)
    gaussians.requires_grad_(False)
    # gaussians.training_setup(opt)
    render_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "renders")
    render_previous_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "renders_previous")
    render_previous_path_train = os.path.join(dataset.model_path, "train", "ours_{}".format(scene.loaded_iter), "renders_previous")
    gts_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_previous_path, exist_ok=True)
    os.makedirs(render_previous_path_train, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    training_data = scene.getTrainCameras()
    testing_data = scene.getTestCameras()

    # Evaluate pose accuracy
    # if scene.scene_info.w_gt and eval_pose:
    #     W2C = []
    #     ref_W2C = []
    #     for c in training_data:
    #         W2C.append(c.world_view_transform.T.cpu().numpy())
    #         ref_W2C.append(c.ref_world_view_transform.T.cpu().numpy())
        
    #     W2C = np.stack(W2C)
    #     ref_W2C = np.stack(ref_W2C)
    #     rpe = compute_rpe(ref_W2C, W2C)
    #     ATE = compute_ATE(ref_W2C, W2C)
    #     print("Training Views: ")
    #     print("RPE:", rpe)
    #     print("ATE:", ATE)
    #     with open(os.path.join(scene.model_path, f"camera/iteration_{scene.loaded_iter}/pose.json"), 'w') as f:
    #         json.dump({"rpe": list(rpe), "ate": float(ATE)}, f)

    fused_data = sorted(
        list(map(lambda x: ('training', x), training_data)) + \
        list(map(lambda x: ('testing', x), testing_data)), key=lambda x: x[1].image_name)
    print("Fused data: ", fused_data)
    # Fused data:  [('training', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), 
    # ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), 
    # ('training', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), 
    # ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), ('testing', Camera()), ('training', Camera())]
    
    
    testing_idx_to_fused_idx = {}
    for testing_ptr in range(len(testing_data)):
        testing_idx_to_fused_idx[testing_ptr] = list(map(lambda x: x[1].image_name, fused_data)).index(testing_data[testing_ptr].image_name)
    print("Testing idx to fused idx: ", testing_idx_to_fused_idx)
    # Testing idx to fused idx:  {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18}
    psnr_before_init_values = []
    psnr_optimizer_values = []
    tensorboard_path = os.path.join(dataset.model_path, "test", "psnr_experiment")
    os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    for train_idx, train_view in tqdm(enumerate(training_data), total=len(training_data)):
        camera_pose = get_tensor_from_camera(train_view.world_view_transform.transpose(0, 1))
        # camera_tensor_T = camera_pose[-3:].requires_grad_()
        # camera_tensor_q = camera_pose[:4].requires_grad_()
        train_out_before_init = render(train_view, gaussians, pipe, background, camera_pose=camera_pose)
        psnr_before_init = psnr(train_view.original_image, train_out_before_init['render'].clamp(0., 1.)).mean()
        print("psnr_before_init:", psnr_before_init)
        torchvision.utils.save_image(train_out_before_init['render'], os.path.join(render_previous_path_train, "{0:05d}".format(train_view.uid) + ".png"))

    for test_idx, test_view in tqdm(enumerate(testing_data), total=len(testing_data)):
        # Initialize from previous camera
        # print("Testing idx: ", test_idx)   # 0
        # print("Testing view: ", test_view)  # DifferentiableCamera_eval()+
        # print("222222222:", fused_data[testing_idx_to_fused_idx[test_idx] - 1][1])
        # if test_idx == 0:
        # if test_idx ==0 or test_idx == 98 or test_idx == 197:
        if test_idx ==0:
            view = fused_data[testing_idx_to_fused_idx[test_idx] - 1][1]
            test_view.world_view_transform = view.world_view_transform
            camera_pose = get_tensor_from_camera(test_view.world_view_transform.transpose(0, 1))  # quar, t → R, T
        else:
            # camera_pose = torch.cat([camera_tensor_q, camera_tensor_T])
            camera_pose = camera_pose_next
            # camera_pose = get_tensor_from_camera(test_view.world_view_transform.transpose(0, 1))  # quar, t → R, T
        # camera_pose = camera_pose_next
        # test_view.init_(fused_data[testing_idx_to_fused_idx[test_idx] - 1][1])
        # test_view.cam_requires_grad_(True)
        # optimizer = torch.optim.Adam([                            # 其实就是更新test_view的R和T，不用更新内参；  因此self.test_view一定得修改，使得R和T可微
        #     {"params": [test_view.quaternion], "lr": 1e-3, "name": "rotation"},
        #     {"params": [test_view.T], "lr": 1e-2, "name": "translation"}
        # ], lr=0.0, maximize=False)

        # 现在需要做的是，把前一帧的pose 给后一帧做初始化，  instant-splat是通过view.world_view_transform转化得到的camera_pose(RT)
        # 所以是不是只需要把前帧的world_view_transform给后帧做初始化就可以了？

        # a = fused_data[testing_idx_to_fused_idx[test_idx] - 1][1]


        #
        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [camera_tensor_T],
                    "lr": 0.0003,
                },
                {
                    "params": [camera_tensor_q],
                    "lr": 0.0001,
                },
            ]
        )

        # print("optimizer: ", optimizer)
        # if test_idx > 82:
        #     print("test_idx===:", test_idx)

        # tensor1 = test_view.quaternion.data
        # tensor2 = test_view.T
        # concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)
        # concatenated_param = torch.nn.Parameter(concatenated_tensor)
        test_out_before_init = render(test_view, gaussians, pipe, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
        psnr_before_init = psnr(test_view.original_image, test_out_before_init['render'].clamp(0., 1.)).mean()
        psnr_before_init_values.append(psnr_before_init.item())
        print("psnr_before_init_values===:", psnr_before_init_values)
        # print("psnr_before_init: ", psnr_before_init)

        torchvision.utils.save_image(test_out_before_init['render'], os.path.join(render_previous_path, "{0:05d}".format(test_view.uid) + ".png"))

        # psnr_s = []
        for _ in range(200):
            optimizer.zero_grad()

            # test_out = render_eval(test_view, gaussians, pipe, background, camera_pose=concatenated_param)
            test_out = render(test_view, gaussians, pipe, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
            # test_out = render_eval(test_view, gaussians, pipe, background, camera_pose=torch.cat(test_view.quaternion, test_view.T))
            # test_out = render_eval(test_view, gaussians, pipe, background)
            # test_out = renderApprSurface(test_view, gaussians, pipe, background)
            
            # kp0, kp1 = predict_hierarchical_correspondence(get_features = {Tensor: (18432, 16, 3)} tensor([[[ 0.6278,  0.2699, -0.2466],\n         [ 0.0000,  0.0000,  0.0000],\n         [ 0.0000,  0.0000,  0.0000],\n         ......    [ 0.0000,  0.0000,  0.0000],\n         [ 0.0000,  0.0000,  0.0000],\n         [ 0.0000,  0.0000,  0.0000]]], device='cuda:0')... View
            #     test_view.image, 
            #     test_out["render"].clamp(0., 1.), 
            #     threshold=0.5
            # )

            # xy0 = kp0 / 2 + .5
            # xy1 = F.grid_sample(norm_xy(test_out["xy"])[None], 
            #         kp1[None, None], mode='bilinear', align_corners=False).reshape(2, -1).permute(1, 0)
            
            # loss = torch.nn.L1Loss()(test_view.image, test_out["render"].clamp(0, 1)) + \
            #     1e2 * torch.nn.L1Loss()(xy0, xy1)
            loss = torch.nn.L1Loss()(test_view.original_image, test_out["render"].clamp(0, 1))
            # print(loss)
            loss.backward()
            # print("test_view.quaternion.requires_grad: ", test_view.quaternion.requires_grad)
            # print("test_view.T.requires_grad: ", test_view.T.requires_grad)
            # print("camera_tensor_T.requires_grad: ", camera_tensor_T.requires_grad)
            # print("camera_tensor_T.grad: ", camera_tensor_T.grad)
            # print("test_view.T.grad: ", test_view.T.grad)
            # print("self._opacity.required_gard: ", gaussians._opacity.requires_grad)
            # print("self.gaussians._opacity.grad: ", gaussians._opacity.grad)
            optimizer.step()
            # psnr_s.append(psnr(test_view.image, test_out["render"].clamp(0., 1.)).mean().detach().item())
        # test_view.cam_requires_grad_(False)
        optimizer.zero_grad()

        # plt.plot(psnr_s)
        # plt.show()
        with torch.no_grad():
            # test_out = renderApprSurface(test_view, gaussians, pipe, background)
            # test_out = render_eval(test_view, gaussians, pipe, background)
            test_out = render(test_view, gaussians, pipe, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
            psnr_optimizer = psnr(test_view.original_image, test_out['render'].clamp(0., 1.)).mean()
            psnr_optimizer_values.append(psnr_optimizer.item())

            tqdm.write(f"{test_view.uid}: {psnr(test_view.original_image, test_out['render'].clamp(0., 1.)).mean()}")
            with open("psnr_test.txt", "a") as file:file.write(f"[ITER {(test_view.uid)}] Evaluating: PSNR {psnr_optimizer}\n")
            # plt.imshow(np.array(render_tensor([test_view.image, test_out['render'].clamp(0., 1.)])))
            # plt.show()
            camera_pose_next = torch.cat([camera_tensor_q, camera_tensor_T])
        gt = test_view.original_image[0:3, :, :]
        torchvision.utils.save_image(test_out['render'], os.path.join(render_path, "{0:05d}".format(test_view.uid) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(test_view.uid) + ".png"))
        # writer.add_scalar('PSNR/before_init', psnr_before_init, test_idx)
        # writer.add_scalar('PSNR/optimizer', psnr_optimizer, test_idx)
        writer.add_scalars('PSNR', {'Before Init': psnr_before_init, 'Optimizer': psnr_optimizer}, test_idx)

    scene.save(scene.loaded_iter, skip_test=False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluating script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--eval_pose", action='store_true', help="Evaluate the pose metrics.")
    args = parser.parse_args(sys.argv[1:])
    print("Registering testing views of " + args.model_path)
    register(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.eval_pose)

    # All done
    print("Register complete.")
