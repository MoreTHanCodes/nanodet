import torch
import torch.nn as nn

from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.scale import Scale


class SimpleConvHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channel,
        width_branch_size=1,
        height_branch_size=1,
        feat_channels=256,
        stacked_convs=4,
        strides=[8, 16, 32],
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        activation="LeakyReLU",
        reg_max=16,
        **kwargs
    ):
        super(SimpleConvHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        assert width_branch_size >= 1
        assert height_branch_size >= 1
        num_output_branches = width_branch_size * height_branch_size
        self.width_branch_size = width_branch_size
        self.height_branch_size = height_branch_size
        self.num_ouput_branches = num_output_branches
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.reg_max = reg_max

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )

        for h in range(self.height_branch_size):
            for w in range(self.width_branch_size):
                gfl_cls = nn.Conv2d(
                    self.feat_channels, self.cls_out_channels, 3, padding=1
                )
                gfl_reg = nn.Conv2d(
                    self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1
                )
                setattr(self, f"gfl_cls_h{h}_w{w}", gfl_cls)
                setattr(self, f"gfl_reg_h{h}_w{w}", gfl_reg)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        for h in range(self.height_branch_size):
            for w in range(self.width_branch_size):
                gfl_cls = getattr(self, f"gfl_cls_h{h}_w{w}")
                gfl_reg = getattr(self, f"gfl_reg_h{h}_w{w}")
                normal_init(gfl_cls, std=0.01, bias=bias_cls)
                normal_init(gfl_reg, std=0.01)

    def forward(self, feats):
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            branch_outputs = []
            for h in range(self.height_branch_size):
                for w in range(self.width_branch_size):
                    gfl_cls = getattr(self, f"gfl_cls_h{h}_w{w}")
                    gfl_reg = getattr(self, f"gfl_reg_h{h}_w{w}")
                    cls_score = gfl_cls(cls_feat)
                    bbox_pred = scale(gfl_reg(reg_feat)).float()
                    output = torch.cat([cls_score, bbox_pred], dim=1)
                    branch_outputs.append(output.flatten(start_dim=2))
            branch_outputs = torch.stack(branch_outputs, dim=1) # [B, N, C, Hi * Wi]
            outputs.append(branch_outputs)
        outputs = torch.cat(outputs, dim=3).permute(0, 1, 3, 2) # [B, N, \sum(Hi * Wi), C]
        return outputs
