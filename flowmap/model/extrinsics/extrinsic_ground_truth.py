from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from ..projection import get_extrinsics
from .extrinsics import Extrinsics



@dataclass
class ExtrinsicsGroundTruthCfg:
    name: Literal["ground_truth"]


class ExtrinsicsGroundTruth(Extrinsics[ExtrinsicsGroundTruthCfg]):
    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        surfaces: Float[Tensor, "batch frame height width 3"],
    ) -> Float[Tensor, "batch frame 4 4"]:
        
        # Just return the ground-truth extrinsics.
        return batch.extrinsics
        
        
