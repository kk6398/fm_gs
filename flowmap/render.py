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

import json
import shutil
from pathlib import Path
from time import time
import os
from os import makedirs
import torchvision
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from flowmap.utils.general_utils import safe_state
from flowmap.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

# # import sys
# from .scene.dataset_readers import readColmapSceneInfo
# from .utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, camera_from_camInfos_selection
# import random
# from random import randint
from flowmap.model.render import render, render_eval
from typing import Any, List, Tuple, Union

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)