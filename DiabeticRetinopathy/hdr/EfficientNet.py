"""
EfficientNet-Bx from scratch (B3 defaults) + linear stochastic-depth schedule.
Author: you
"""

import torch
from torch import nn
from math import ceil
from typing import List


# ───────────────────────────── layers ──────────────────────────────
class ConvBnAct(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1,
                 padding=0, groups=1, bias=False, bn=True, act=True):
        super().__init__()
        # ----------- FIXED: use keyword args -----------
        self.conv = nn.Conv2d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,      # now really “groups”
            bias=bias
        )
        # -----------------------------------------------
        self.bn  = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, n_in, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_in, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, n_in, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class StochasticDepth(nn.Module):
    def __init__(self, survival_prob: float):
        super().__init__()
        self.p = survival_prob

    def forward(self, x):
        if (not self.training) or self.p == 1.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        return x.div(self.p) * keep


class MBConvN(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1,
                 expansion_factor=6, reduction=4, survival_prob=0.8):
        super().__init__()
        self.use_res = stride == 1 and n_in == n_out
        mid = int(n_in * expansion_factor)
        pad = (kernel_size - 1) // 2
        red = max(1, n_in // reduction)

        self.expand = nn.Identity() if expansion_factor == 1 else \
                      ConvBnAct(n_in, mid, 1)
        self.depthwise = ConvBnAct(mid, mid, kernel_size, stride,
                                   pad, groups=mid)
        self.se   = SqueezeExcitation(mid, red)
        self.point = ConvBnAct(mid, n_out, 1, act=False)
        self.drop  = StochasticDepth(survival_prob)

    def forward(self, x):
        out = self.point(self.se(self.depthwise(self.expand(x))))
        if self.use_res:
            out = self.drop(out) + x
        return out


# ───────────────────── helper: linear SD schedule ───────────────────
def sd_prob(idx: int, total: int, pl: float = 0.2) -> float:
    """Return survival probability for block #idx (0-based)."""
    return 1.0 - pl * idx / (total - 1)


# ─────────────────────────── EfficientNet ───────────────────────────
class EfficientNet(nn.Module):
    """
    width_mult, depth_mult default to B3 (1.2, 1.4).
    """
    def __init__(self, width_mult: float = 1.2, depth_mult: float = 1.4,
                 dropout_rate: float = 0.30, num_classes: int = 5):
        super().__init__()

        # stem
        out_channels = _round_filters(32, width_mult)
        layers: List[nn.Module] = [ConvBnAct(3, out_channels, 3, 2, 1)]
        in_ch = out_channels

        # MBConv config (EffNet-B0 baseline)
        cfg = dict(
            k=[3, 3, 5, 3, 5, 5, 3],
            exp=[1, 6, 6, 6, 6, 6, 6],
            c=[16, 24, 40, 80, 112, 192, 320],
            n=[1, 2, 2, 3, 3, 4, 1],
            s=[1, 2, 2, 2, 1, 2, 1]
        )
        cfg["c"] = [_round_filters(c, width_mult) for c in cfg["c"]]
        cfg["n"] = [_round_repeats(n, depth_mult) for n in cfg["n"]]
        total_blocks = sum(cfg["n"])
        block_id = 0

        for i in range(len(cfg["c"])):
            for r in range(cfg["n"][i]):
                layers.append(
                    MBConvN(
                        n_in=in_ch if r == 0 else cfg["c"][i],
                        n_out=cfg["c"][i],
                        kernel_size=cfg["k"][i],
                        stride=cfg["s"][i] if r == 0 else 1,
                        expansion_factor=cfg["exp"][i],
                        survival_prob=sd_prob(block_id, total_blocks, 0.2)
                    )
                )
                in_ch = cfg["c"][i]
                block_id += 1

        # head
        last_ch = _round_filters(1280, width_mult)
        layers.append(ConvBnAct(in_ch, last_ch, 1))

        self.features = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_ch, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.flatten(1))

    # ── utils ──
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)


def _round_filters(c, width_mult):
    return int(4 * ceil(c * width_mult / 4))


def _round_repeats(r, depth_mult):
    return int(ceil(r * depth_mult))


# ───────────────── factory with ImageNet warm-start ──────────────────
def build_efficientnet_b3(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    model = EfficientNet(num_classes=num_classes)  # B3 scaling is default above
    if pretrained:
        # load matching weights from RWightman repo
        state = torch.hub.load('rwightman/gen-efficientnet-pytorch',
                               'efficientnet_b3',
                               pretrained=True).state_dict()
        msd = model.state_dict()
        msd.update({k: v for k, v in state.items() if k in msd})
        model.load_state_dict(msd)
    return model
