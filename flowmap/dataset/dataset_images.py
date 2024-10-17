from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from math import floor
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon
from .types import Stage


@dataclass
class DatasetImagesCfg(DatasetCfgCommon):
    name: Literal["images"]
    root: Path


FAKE_REPETITIONS = 1000


class DatasetImages(Dataset):
    def __init__(
        self,
        cfg: DatasetImagesCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_sampler = frame_sampler

        # Fixed image shapes are intended for pretraining, but this dataset is intended
        # for overfitting.
        assert cfg.image_shape is None

        # Load the images.
        self.frame_paths = tuple(sorted(cfg.root.iterdir()))
        self.images = [tf.ToTensor()(Image.open(path))[:3] for path in self.frame_paths]   # tf.ToTensor(): 图像的像素值从[0, 255]缩放到[0.0, 1.0]，以及从HxWxC的格式转换为CxHxW

    def __getitem__(self, index: int):
        # Run the frame sampler.
        num_frames = len(self.images)

        # train_view = []
        # length = 12
        #
        # interval = floor((num_frames - length) / (length - 1))   # (200-12)/(12-1)=17
        # for i in range(0, num_frames):
        #     if i % (interval + 1) == 0:
        #         train_view.append(i)
        # train_view[-1] = num_frames - 1  # 强制让最后一个值为总数200

        indices = self.frame_sampler.sample(num_frames, torch.device("cpu"))
        # indices = torch.tensor(train_view)
        return {
            "videos": torch.stack([self.images[i] for i in indices]),
            # "videos": torch.stack([self.images[i] for i in train_view]),
            "indices": indices,
            "scenes": self.cfg.root.stem,
            "datasets": "images",
            "frame_paths": [self.frame_paths[i] for i in indices],
            # "frame_paths": [self.frame_paths[i] for i in train_view],
        }

    def __len__(self) -> int:
        # Return a much larger length for compatibility with PyTorch Lightning.
        return FAKE_REPETITIONS
