import math
from turtle import forward

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from ..module.scale import Scale
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import reduce_mean


class OffsetDistProj(nn.Module):
    def __init__(
        self,
        num_offsets,
        reg_start,
        reg_end,
        reg_max
    ):
        super(OffsetDistProj, self).__init__()

        self.num_offsets = num_offsets
        self.reg_start = reg_start
        self.reg_end = reg_end
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(self.reg_start, self.reg_end, self.reg_max + 1)
        )
    
    def forward(self, x):
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], self.num_offsets, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], self.num_offsets)
        return x


class LiteAnchorBasedHead(nn.Module):
    """Anchor-based detection head for task with pre-defined 3D model.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        width_branch_size (int): Multi-view branches size along width axis.
            Default: 1.
        height_branch_size (int): Multi-view branches size along height axis.
            Default: 1.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        kpt_offset_reg_start (float): The minimal value of the offset between
            the feature map grid and the bbox keypoint. Default: -7.0.
        kpt_offset_reg_end (float): The maximal value of the offset between
            the feature map grid and the bbox keypoint. Default: 7.0.
        kpt_offset_reg_max (int): The maximal value of the discrete set to
            represent the keypoint offset. Default: 14.
        bbox_offset_reg_start (float): The minimal value of the distance offset
            from keypoint to bbox. Default: 0.0.
        bbox_offset_reg_end (float): The maximal value of the distance offset
            from keypoint to bbox. Default: 7.0.
        bbox_offset_reg_max (int): The maximal value of the discrete set to
            represent the bbox offset. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        width_branch_size=1,
        height_branch_size=1,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        kpt_offset_reg_start=-7.0,
        kpt_offset_reg_end=7.0,
        kpt_offset_reg_max=14,
        bbox_offset_reg_start=0.0,
        bbox_offset_reg_end=9.0,
        bbox_offset_reg_max=9,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(LiteAnchorBasedHead, self).__init__()
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
        self.kernel_size = kernel_size
        self.strides = strides
        self.kpt_offset_reg_start = kpt_offset_reg_start
        self.kpt_offset_reg_end = kpt_offset_reg_end
        self.kpt_offset_reg_max = kpt_offset_reg_max
        self.bbox_offset_reg_start = bbox_offset_reg_start
        self.bbox_offset_reg_end = bbox_offset_reg_end
        self.bbox_offset_reg_max = bbox_offset_reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.kpt_offset_dist_project = OffsetDistProj(
            num_offsets=2,
            reg_start=kpt_offset_reg_start,
            reg_end=kpt_offset_reg_end,
            reg_max=kpt_offset_reg_max
        )
        self.bbox_offset_dist_project = OffsetDistProj(
            num_offsets=4,
            reg_start=bbox_offset_reg_start,
            reg_end=bbox_offset_reg_end,
            reg_max=bbox_offset_reg_max
        )

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        
        for h in range(self.height_branch_size):
            for w in range(self.width_branch_size):
                gfl_cls = nn.ModuleList(
                    [
                        nn.Conv2d(
                            self.feat_channels,
                            self.num_classes,
                            1,
                            padding=0,
                        )
                        for _ in self.strides
                    ]
                )
                gfl_reg = nn.ModuleList(
                    [
                        nn.Conv2d(
                            self.feat_channels,
                            2 * (self.kpt_offset_reg_max + 1) + 4 * (self.bbox_offset_reg_max + 1),
                            1,
                            padding=0,
                        )
                        for _ in self.strides
                    ]
                )
                setattr(self, f"gfl_cls_h{h}_w{w}", gfl_cls)
                setattr(self, f"gfl_reg_h{h}_w{w}", gfl_reg)

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            reg_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs, reg_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for h in range(self.height_branch_size):
            for w in range(self.width_branch_size):
                gfl_cls = getattr(self, f"gfl_cls_h{h}_w{w}")
                gfl_reg = getattr(self, f"gfl_reg_h{h}_w{w}")
                for i in range(len(self.strides)):
                    normal_init(gfl_cls[i], std=0.01, bias=bias_cls)
                    normal_init(gfl_reg[i], std=0.01)
        print("Finish initialize multi-branches custom detection Head.")

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for i, (feat, cls_convs, reg_convs) in enumerate(zip(
            feats,
            self.cls_convs,
            self.reg_convs
        )):
            cls_feat = feat
            reg_feat = feat
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            branch_outputs = []
            for h in range(self.height_branch_size):
                for w in range(self.width_branch_size):
                    gfl_cls = getattr(self, f"gfl_cls_h{h}_w{w}")
                    gfl_reg = getattr(self, f"gfl_reg_h{h}_w{w}")
                    cls_score = gfl_cls[i](cls_feat)
                    bbox_pred = gfl_reg[i](reg_feat)
                    output = torch.cat([cls_score, bbox_pred], dim=1)
                    branch_outputs.append(output.flatten(start_dim=2))
            branch_outputs = torch.stack(branch_outputs, dim=1) # [B, N, C, Hi * Wi]
            outputs.append(branch_outputs)
        outputs = torch.cat(outputs, dim=3).permute(0, 1, 3, 2) # [B, N, \sum(Hi * Wi), C]
        return outputs

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        gt_keypoints = gt_meta["gt_keypoints"]
        device = preds.device
        batch_size = preds.shape[0]
        input_height, input_width = gt_meta["img"].shape[2:]
        # NOTE: DO NOT consider width/height branch size in input_shape
        input_shape = (input_height, input_width)
        input_height = input_height // self.height_branch_size
        input_width = input_width // self.width_branch_size

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]

        total_loss = 0
        total_loss_states = None
        for h in range(self.height_branch_size):
            for w in range(self.width_branch_size):
                n = h * self.width_branch_size + w
                branch_preds = preds[:, n]
                cls_preds, kpt_offset_reg_preds, bbox_offset_reg_preds = branch_preds.split(
                    [self.num_classes, 2 * (self.kpt_offset_reg_max + 1), 4 * (self.bbox_offset_reg_max + 1)], dim=-1
                )
                center_priors = torch.cat(mlvl_center_priors, dim=1)
                center_priors[..., 0] += (torch.ones_like(center_priors[..., 0]) * input_width * w)
                center_priors[..., 1] += (torch.ones_like(center_priors[..., 1]) * input_height * h)

                kpt_offset_preds = self.kpt_offset_dist_project(kpt_offset_reg_preds) * center_priors[..., 2, None]
                bbox_offset_preds = self.bbox_offset_dist_project(bbox_offset_reg_preds) * center_priors[..., 2, None]
                d0 = bbox_offset_preds[..., 0] - kpt_offset_preds[..., 0]
                d1 = bbox_offset_preds[..., 1] - kpt_offset_preds[..., 1]
                d2 = bbox_offset_preds[..., 2] + kpt_offset_preds[..., 0]
                d3 = bbox_offset_preds[..., 3] + kpt_offset_preds[..., 1]
                dis_preds = torch.stack([d0, d1, d2, d3], dim=-1)
                # decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
                decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

                if aux_preds is not None:
                    # use auxiliary head to assign
                    aux_branch_preds = aux_preds[:, n]
                    aux_cls_preds, aux_kpt_reg_preds, aux_bbox_reg_preds = aux_branch_preds.split(
                        [self.num_classes, 2 * (self.kpt_offset_reg_max + 1), 4 * (self.bbox_offset_reg_max + 1)], dim=-1
                    )
                    aux_kpt_preds = self.kpt_offset_dist_project(aux_kpt_reg_preds) * center_priors[..., 2, None]
                    aux_bbox_preds = self.bbox_offset_dist_project(aux_bbox_reg_preds) * center_priors[..., 2, None]
                    d0 = aux_bbox_preds[..., 0] - aux_kpt_preds[..., 0]
                    d1 = aux_bbox_preds[..., 1] - aux_kpt_preds[..., 1]
                    d2 = aux_bbox_preds[..., 2] + aux_kpt_preds[..., 0]
                    d3 = aux_bbox_preds[..., 3] + aux_kpt_preds[..., 1]
                    aux_dis_preds = torch.stack([d0, d1, d2, d3], dim=-1)
                    # aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds, max_shape=input_shape)
                    aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
                    batch_assign_res = multi_apply(
                        self.target_assign_single_img,
                        aux_cls_preds.detach(),
                        center_priors,
                        aux_decoded_bboxes.detach(),
                        gt_bboxes,
                        gt_keypoints,
                        gt_labels,
                    )
                else:
                    # use self prediction to assign
                    batch_assign_res = multi_apply(
                        self.target_assign_single_img,
                        cls_preds.detach(),
                        center_priors,
                        decoded_bboxes.detach(),
                        gt_bboxes,
                        gt_keypoints,
                        gt_labels,
                    )

                loss, loss_states = self._get_loss_from_assign(
                    cls_preds, kpt_offset_reg_preds, bbox_offset_reg_preds, decoded_bboxes, batch_assign_res
                )

                if aux_preds is not None:
                    aux_loss, aux_loss_states = self._get_loss_from_assign(
                        aux_cls_preds, aux_kpt_reg_preds, aux_bbox_reg_preds, aux_decoded_bboxes, batch_assign_res
                    )
                    loss = loss + aux_loss
                    for k, v in aux_loss_states.items():
                        loss_states["aux_" + k] = v
                
                total_loss += loss
                if total_loss_states is None:
                    total_loss_states = loss_states
                else:
                    for k, v in loss_states.items():
                        total_loss_states[k] += v

        return total_loss, total_loss_states

    def _get_loss_from_assign(self, cls_preds, kpt_reg_preds, bbox_reg_preds, decoded_bboxes, assign):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, keypoint_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        kpt_reg_preds = kpt_reg_preds.reshape(-1, 2 * (self.kpt_offset_reg_max + 1))
        bbox_reg_preds = bbox_reg_preds.reshape(-1, 4 * (self.bbox_offset_reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                bbox_reg_preds[pos_inds].reshape(-1, self.bbox_offset_reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )

            keypoint_targets = torch.cat(keypoint_targets, dim=0)
            loss_dfl += self.loss_dfl(
                kpt_reg_preds[pos_inds].reshape(-1, self.kpt_offset_reg_max + 1),
                keypoint_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 2).reshape(-1),
                avg_factor=2.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = bbox_reg_preds.sum() * 0 + kpt_reg_preds.sum() * 0
            loss_dfl = bbox_reg_preds.sum() * 0 + kpt_reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_keypoints, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_keypoints (Tensor): Ground truth keypoints of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)
        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_keypoints = torch.from_numpy(gt_keypoints).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        gt_keypoints = gt_keypoints.to(decoded_bboxes.dtype)

        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        # keypoint_targets = torch.zeros_like(center_priors)
        keypoint_targets = center_priors.new_zeros((num_priors, 2), dtype=torch.float)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        # No target
        if num_gts == 0:
            # return labels, label_scores, bbox_targets, dist_targets, 0
            return labels, label_scores, bbox_targets, dist_targets, keypoint_targets, 0

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_gt_keypoints, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes, gt_keypoints
        )
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            keypoint_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_keypoints)[..., 2:]
                / center_priors[pos_inds, None, 2]
            )
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets[pos_inds, 0] += keypoint_targets[pos_inds, 0]
            dist_targets[pos_inds, 1] += keypoint_targets[pos_inds, 1]
            dist_targets[pos_inds, 2] -= keypoint_targets[pos_inds, 0]
            dist_targets[pos_inds, 3] -= keypoint_targets[pos_inds, 1]
            keypoint_targets = keypoint_targets.clamp(
                min=self.kpt_offset_reg_start,
                max=self.kpt_offset_reg_end
            )
            dist_targets = dist_targets.clamp(
                min=self.bbox_offset_reg_start,
                max=self.bbox_offset_reg_end
            )
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
        # return (
        #     labels,
        #     label_scores,
        #     bbox_targets,
        #     dist_targets,
        #     num_pos_per_img,
        # )
        return (
            labels,
            label_scores,
            bbox_targets,
            dist_targets,
            keypoint_targets,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes, gt_keypoints):
        """Sample positive and negative bboxes."""
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
            pos_gt_keypoints = torch.empty_like(gt_keypoints).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
                gt_keypoints = gt_keypoints.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
            pos_gt_keypoints = gt_keypoints[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_gt_keypoints, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 2 * (self.kpt_offset_reg_max + 1) + 4 * (self.bbox_offset_reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            det_bboxes[:, 5:7] = warp_boxes(
                det_bboxes[:, 5:7], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                        det_bboxes[inds, 5:7].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 2 * (kpt_offset_regmax + 1) + 4 * (bbox_offset_regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        # NOTE: DO NOT consider width/height branch size in input_shape
        input_shape = (input_height, input_width)
        input_height = input_height // self.height_branch_size
        input_width = input_width // self.width_branch_size

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]

        batch_branches_bboxes = []
        batch_branches_scores = []
        batch_branches_keypoints = []
        for h in range(self.height_branch_size):
            for w in range(self.width_branch_size):
                n = h * self.width_branch_size + w
                branch_reg_preds = reg_preds[:, n]
                branch_cls_preds = cls_preds[:, n]
                kpt_offset_reg_preds, bbox_offset_reg_preds = branch_reg_preds.split(
                    [2 * (self.kpt_offset_reg_max + 1), 4 * (self.bbox_offset_reg_max + 1)], dim=-1
                )
                center_priors = torch.cat(mlvl_center_priors, dim=1)
                center_priors[..., 0] += (torch.ones_like(center_priors[..., 0]) * input_width * w)
                center_priors[..., 1] += (torch.ones_like(center_priors[..., 1]) * input_height * h)

                kpt_offset_preds = self.kpt_offset_dist_project(kpt_offset_reg_preds) * center_priors[..., 2, None]
                bbox_offset_preds = self.bbox_offset_dist_project(bbox_offset_reg_preds) * center_priors[..., 2, None]
                d0 = bbox_offset_preds[..., 0] - kpt_offset_preds[..., 0]
                d1 = bbox_offset_preds[..., 1] - kpt_offset_preds[..., 1]
                d2 = bbox_offset_preds[..., 2] + kpt_offset_preds[..., 0]
                d3 = bbox_offset_preds[..., 3] + kpt_offset_preds[..., 1]
                branch_dis_preds = torch.stack([d0, d1, d2, d3], dim=-1)
                branch_kpt_preds = center_priors[..., :2] + kpt_offset_preds
                branch_kpt_preds[..., 0] = branch_kpt_preds[..., 0].clamp(min=0, max=input_shape[1])
                branch_kpt_preds[..., 1] = branch_kpt_preds[..., 1].clamp(min=0, max=input_shape[0])
                branch_bboxes = distance2bbox(center_priors[..., :2], branch_dis_preds, max_shape=input_shape)
                branch_scores = branch_cls_preds.sigmoid()
                batch_branches_bboxes.append(branch_bboxes)
                batch_branches_scores.append(branch_scores)
                batch_branches_keypoints.append(branch_kpt_preds)
        batch_branches_bboxes = torch.stack(batch_branches_bboxes, dim=1)
        batch_branches_scores = torch.stack(batch_branches_scores, dim=1)
        batch_branches_keypoints = torch.stack(batch_branches_keypoints, dim=1)

        result_list = []
        for b in range(batch_size):
            branches_bboxes, branches_labels = [], []
            for h in range(self.height_branch_size):
                for w in range(self.width_branch_size):
                    n = h * self.width_branch_size + w
                    # add a dummy background class at the end of all labels
                    # same with mmdetection2.0
                    score, bbox, keypoints = batch_branches_scores[b][n], batch_branches_bboxes[b][n], batch_branches_keypoints[b][n]
                    padding = score.new_zeros(score.shape[0], 1)
                    score = torch.cat([score, padding], dim=1)
                    bboxes, labels = multiclass_nms(
                        bbox,
                        score,
                        score_thr=0.05,
                        nms_cfg=dict(type="nms", iou_threshold=0.6),
                        max_num=1, # NOTE: change this value accordingly
                        multi_keypoints=keypoints,
                    )
                    branches_bboxes.append(bboxes)
                    branches_labels.append(labels)
            branches_bboxes = torch.cat(branches_bboxes, dim=0)
            branches_labels = torch.cat(branches_labels, dim=0)
            results = (branches_bboxes, branches_labels)
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _forward_onnx(self, feats):
        outputs = []
        for i, (feat, cls_convs, reg_convs) in enumerate(zip(
            feats,
            self.cls_convs,
            self.reg_convs
        )):
            cls_feat = feat
            reg_feat = feat
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            branch_outputs = []
            for h in range(self.height_branch_size):
                for w in range(self.width_branch_size):
                    gfl_cls = getattr(self, f"gfl_cls_h{h}_w{w}")
                    gfl_reg = getattr(self, f"gfl_reg_h{h}_w{w}")
                    cls_score = gfl_cls[i](cls_feat).sigmoid()
                    bbox_pred = gfl_reg[i](reg_feat)
                    output = torch.cat([cls_score, bbox_pred], dim=1)
                    branch_outputs.append(output.flatten(start_dim=2))
            branch_outputs = torch.stack(branch_outputs, dim=1) # [B, N, C, Hi * Wi]
            outputs.append(branch_outputs)
        outputs = torch.cat(outputs, dim=3).permute(0, 1, 3, 2) # [B, N, \sum(Hi * Wi), C]
        return outputs

    # def _forward_onnx(self, feats):
    #     """only used for onnx export"""
    #     outputs = []
    #     for i, (feat, cls_convs, reg_convs) in enumerate(zip(
    #         feats,
    #         self.cls_convs,
    #         self.reg_convs
    #     )):
    #         cls_feat = feat
    #         reg_feat = feat
    #         for cls_conv in cls_convs:
    #             cls_feat = cls_conv(cls_feat)
    #         for reg_conv in reg_convs:
    #             reg_feat = reg_conv(reg_feat)
    #         branch_outputs = []
    #         for h in range(self.height_branch_size):
    #             for w in range(self.width_branch_size):
    #                 gfl_cls = getattr(self, f"gfl_cls_h{h}_w{w}")
    #                 gfl_reg = getattr(self, f"gfl_reg_h{h}_w{w}")
    #                 cls_score = gfl_cls[i](cls_feat)
    #                 bbox_pred = gfl_reg[i](reg_feat)
    #                 output = torch.cat([cls_score, bbox_pred], dim=1)
    #                 branch_outputs.append(output.flatten(start_dim=2))
    #         branch_outputs = torch.stack(branch_outputs, dim=1) # [B, N, C, Hi * Wi]
    #         outputs.append(branch_outputs)
    #     outputs = torch.cat(outputs, dim=3).permute(0, 1, 3, 2) # [B, N, \sum(Hi * Wi), C]
    #     return outputs


class AnchorBasedHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channel,
        width_branch_size=1,
        height_branch_size=1,
        feat_channels=256,
        stacked_convs=4,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        kpt_offset_reg_start=-7.0,
        kpt_offset_reg_end=7.0,
        kpt_offset_reg_max=14,
        bbox_offset_reg_start=0.0,
        bbox_offset_reg_end=9.0,
        bbox_offset_reg_max=9,
        activation="LeakyReLU",
        **kwargs
    ):
        super(AnchorBasedHead, self).__init__()
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
        self.kernel_size = kernel_size
        self.strides = strides
        self.kpt_offset_reg_start = kpt_offset_reg_start
        self.kpt_offset_reg_end = kpt_offset_reg_end
        self.kpt_offset_reg_max = kpt_offset_reg_max
        self.bbox_offset_reg_start = bbox_offset_reg_start
        self.bbox_offset_reg_end = bbox_offset_reg_end
        self.bbox_offset_reg_max = bbox_offset_reg_max
        self.activation = activation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
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
                    self.feat_channels,
                    self.cls_out_channels,
                    3,
                    padding=1
                )
                gfl_reg = nn.Conv2d(
                    self.feat_channels,
                    2 * (self.kpt_offset_reg_max + 1) + 4 * (self.bbox_offset_reg_max + 1),
                    3,
                    padding=1
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