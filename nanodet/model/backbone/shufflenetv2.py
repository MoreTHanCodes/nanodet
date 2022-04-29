import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..module.activation import act_layers

model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",  # noqa: E501
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",  # noqa: E501
    "shufflenetv2_1.5x": None,
    "shufflenetv2_2.0x": None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="1.5x",
        out_stages=(2, 3, 4),
        width_batch_size=1,
        height_batch_size=1,
        image_channels=3,
        with_last_conv=False,
        kernal_size=3,
        activation="ReLU",
        pretrain=True,
    ):
        super(ShuffleNetV2, self).__init__()
        # out_stages can only be a subset of (1, 2, 3, 4)
        assert set(out_stages).issubset((1, 2, 3, 4))

        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        self.activation = activation
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        assert width_batch_size >= 1
        assert height_batch_size >= 1
        assert image_channels in [1, 3]
        num_input_images = width_batch_size * height_batch_size
        self.width_batch_size = width_batch_size
        self.height_batch_size = height_batch_size
        self.num_input_images = num_input_images
        self.image_channels = image_channels
        # input_channels = image_channels
        # output_channels = self._stage_out_channels[0] // num_input_images
        # assert (output_channels * num_input_images) == self._stage_out_channels[0]
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     act_layers(activation),
        # )
        # input_channels = self._stage_out_channels[0]
        input_channels = image_channels * num_input_images
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers(activation),
        )
        input_channels = self._stage_out_channels[0]

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, activation=activation
                )
            ]
            for i in range(repeats - 1):
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, activation=activation
                    )
                )
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module("conv5", conv5)
        self._initialize_weights(pretrain)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        BH, BW = self.height_batch_size, self.width_batch_size
        # x_subs = []
        # for bh in range(BH):
        #     for bw in range(BW):
        #         h0 = bh * (H // BH)
        #         h1 = h0 + (H // BH)
        #         w0 = bw * (W // BW)
        #         w1 = w0 + (W // BW)
        #         x_sub = x[:, :, h0:h1, w0:w1]
        #         x_subs.append(self.conv1(x_sub))
        # x = torch.cat(x_subs, dim=1)
        # x = self.maxpool(x)
        x_subs = []
        for bh in range(BH):
            for bw in range(BW):
                h0 = bh * (H // BH)
                h1 = h0 + (H // BH)
                w0 = bw * (W // BW)
                w1 = w0 + (W // BW)
                x_sub = x[:, :, h0:h1, w0:w1]
                x_subs.append(x_sub)
        x = torch.cat(x_subs, dim=1)
        x = self.conv1(x)
        x = self.maxpool(x)

        output = []
        if 1 in self.out_stages:
            output.append(x)
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if pretrain:
            url = model_urls["shufflenetv2_{}".format(self.model_size)]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)
                print("=> loading pretrained model {}".format(url))
                if self.num_input_images == 1 and self.image_channels == 3:
                    pass
                else:
                    _ = pretrained_state_dict.pop("conv1.0.weight")
                    # _ = pretrained_state_dict.pop("conv1.1.weight")
                    # _ = pretrained_state_dict.pop("conv1.1.bias")
                    # _ = pretrained_state_dict.pop("conv1.1.running_mean")
                    # _ = pretrained_state_dict.pop("conv1.1.running_var")
                self.load_state_dict(pretrained_state_dict, strict=False)
