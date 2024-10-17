from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor
from math import floor
from .frame_sampler import FrameSampler


@dataclass
class FrameSamplerOverfitCfg:
    name: Literal["overfit"]
    start: int | None
    num_frames: int | None
    step: int | None


class FrameSamplerOverfit(FrameSampler[FrameSamplerOverfitCfg]):
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
    ) -> Int64[Tensor, " frame"]:
        train_view = []
        start = self.cfg.start or 0
        num_frames = self.cfg.num_frames or num_frames_in_video
        step = self.cfg.step or 1

        if num_frames ==3 or num_frames==6 or num_frames==12:
            total_num_frames = num_frames_in_video
            interval = floor((total_num_frames - self.cfg.num_frames) / (self.cfg.num_frames - 1))  # (200-12)/(12-1)=17
            for i in range(0, total_num_frames):
                if i % (interval + 1) == 0:
                    train_view.append(i)
            train_view[-1] = total_num_frames - 1  # 强制让最后一个值为总数200
            return torch.tensor(train_view)
        else:
            return torch.arange(
                start,
                start + num_frames * step,
                step,
                device=device,
            )
