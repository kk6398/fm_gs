from .loss import Loss
from .loss_flow import LossFlow, LossFlowCfg
from .loss_tracking import LossTracking, LossTrackingCfg
# #
LOSSES = {                   # 这两者都用了
    "flow": LossFlow, 
    "tracking": LossTracking,
}

LossCfg = LossFlowCfg | LossTrackingCfg


def get_losses(cfgs: list[LossCfg]) -> list[Loss]:
    return [LOSSES[cfg.name](cfg) for cfg in cfgs]
