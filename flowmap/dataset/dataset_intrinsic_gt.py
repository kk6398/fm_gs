import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from math import floor

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from ..export.colmap import read_colmap_model
from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon
from .dataset_images import DatasetImages, DatasetImagesCfg
from .types import Stage


@dataclass
class DatasetIntrinsicGTCfg(DatasetCfgCommon):
    name: Literal["intrinsicgt"]
    root: Path
    reorder: bool
    use_image_folder_fallback: bool
    
FAKE_REPETITIONS = 1000
    

class DatasetIntrinsicGT(Dataset):
    def __init__(
        self,
        cfg: DatasetIntrinsicGTCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_sampler = frame_sampler

        # Use the image dataset as a fallback.
        if cfg.use_image_folder_fallback and not (cfg.root / "colmap_train/sparse").exists() and not (cfg.root / "sparse").exists():
            self.fallback = DatasetImages(
                DatasetImagesCfg(
                    self.cfg.image_shape, self.cfg.scene, "images", self.cfg.root
                ),
                stage,
                frame_sampler,
            )
            return
        else:
            self.fallback = None
            
        # Read the COLMAP model.
        if (cfg.root / "colmap_train/sparse").exists():
            colmap_path = Path(os.path.join(cfg.root, "colmap_train/sparse/0"))
        else:
            colmap_path = Path(os.path.join(cfg.root, "sparse/0"))

        self.extrinsics, self.intrinsics, image_names = read_colmap_model(
            colmap_path, reorder=cfg.reorder                                             # "colmap/sparse/0"
        )
        
        # Fixed image shapes are intended for pretraining, but this dataset is intended
        # for overfitting.
        assert cfg.image_shape is None

        ### 需要将self.intrinsics的维度 [N,3,3] 扩展成 [len(all_images), 3, 3]

        # Load the images.
        all_images_path = Path(os.path.join(cfg.root, "images"))
        self.frame_paths = tuple(sorted(all_images_path.iterdir()))

        # self.frame_paths = [cfg.root / "images" / name for name in image_names]
        self.images = [tf.ToTensor()(Image.open(path))[:3] for path in self.frame_paths]
        if self.intrinsics.shape[0] != len(self.images):
            self.intrinsics = self.intrinsics.repeat(len(self.images), 1, 1)

        # self.extrinsics操作，


        num_fewshot_frames = 12
        train_view = []
        total_num_frames = len(self.images)       # 200
        interval = floor((total_num_frames - num_fewshot_frames) / (num_fewshot_frames - 1))  # (200-12)/(12-1)=17
        for i in range(0, total_num_frames):
            if i % (interval + 1) == 0:
                train_view.append(i)
        train_view[-1] = total_num_frames - 1  # 强制让最后一个值为总数200

        # 创建一个新的张量，它的维度与 self.extrinsics 相同
        new_extrinsics = torch.zeros((len(self.images), 4, 4))

        # 使用 train_view 中的索引来填充新的张量
        for i in range(0, self.extrinsics.shape[0]):   # i=0
            # print("i===", i)
            if i != (self.extrinsics.shape[0] -1):
                for j in range(train_view[i], train_view[i+1]):   # j=0
                    new_extrinsics[j, :, :] = self.extrinsics[i, :, :]
            else:
                j = train_view[i]
                new_extrinsics[j, :, :] = self.extrinsics[i, :, :]

        self.extrinsics = new_extrinsics


        
    def __getitem__(self, index: int):
        if self.fallback is not None:
            return self.fallback.__getitem__(index)

        # Run the frame sampler.
        num_frames = len(self.images)
        indices = self.frame_sampler.sample(num_frames, torch.device("cpu"))

        return {
            "videos": torch.stack([self.images[i] for i in indices]),
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics[indices],
            "indices": indices,
            "scenes": self.cfg.root.stem,
            "datasets": "images",
            "frame_paths": [self.frame_paths[i] for i in indices],
        }

    def __len__(self) -> int:
        # Return a much larger length for compatibility with PyTorch Lightning.
        return FAKE_REPETITIONS