import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms_with_mask
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import numpy as np
INF = 1e8
import time


@HEADS.register_module
class SoloHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(SoloHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        #self.loss_mask = build_loss(loss_mask)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.grid_num=[40,36,24,16,12]
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
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
                    bias=self.norm_cfg is None))
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.solo_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        #use P3-P7 now, will change to P2-P6 later
        self.solo_mask3 = nn.Conv2d(self.feat_channels, 40*40, 1, padding=0)
        self.solo_mask4 = nn.Conv2d(self.feat_channels, 36*36, 1, padding=0)
        self.solo_mask5 = nn.Conv2d(self.feat_channels, 24*24, 1, padding=0)
        self.solo_mask6 = nn.Conv2d(self.feat_channels, 16*16, 1, padding=0)
        self.solo_mask7 = nn.Conv2d(self.feat_channels, 12*12, 1, padding=0)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.solo_cls, std=0.01, bias=bias_cls)
        normal_init(self.solo_mask3, std=0.01)
        normal_init(self.solo_mask4, std=0.01)
        normal_init(self.solo_mask5, std=0.01)
        normal_init(self.solo_mask6, std=0.01)
        normal_init(self.solo_mask7, std=0.01)

    def forward(self, feats):
        cls_score, mask_feats = multi_apply(self.forward_single, feats, self.scales)
        mask_feats[0] = self.solo_mask3(mask_feats[0])
        mask_feats[1] = self.solo_mask4(mask_feats[1])
        mask_feats[2] = self.solo_mask5(mask_feats[2])
        mask_feats[3] = self.solo_mask6(mask_feats[3])
        mask_feats[4] = self.solo_mask7(mask_feats[4])
        for i in range(5):
            cls_score[i]=F.upsample_bilinear(cls_score[i],(self.grid_num[i],self.grid_num[i]))
        return cls_score, mask_feats

    def forward_single(self, x, scale):
        cls_feat = x
        mask_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.solo_cls(cls_feat)

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        return cls_score, mask_feat

    def dice_loss(self,input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))



    @force_fp32(apply_to=('cls_scores', 'mask_preds'))
    def loss(self,
             cls_scores,
             mask_preds,
             gt_bboxes,
             gt_labels,
             gt_masks,
             category_targets,
             point_ins,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(mask_preds)
        loss_mask = 0
        _,_,b_h,b_w = mask_preds[0].shape
        for i in range(5):
            #cls_scores[i]=F.upsample_bilinear(cls_scores[i],(self.grid_num[i],self.grid_num[i]))
            mask_preds[i] = F.sigmoid(F.upsample_bilinear(mask_preds[i],(b_h,b_w)))
        bound=[self.grid_num[i]**2 for i in range(5)]
        for i in range(1,5):
            bound[i] += bound[i-1]
         
        num_imgs=len(category_targets)
        for i in range(num_imgs):
            _, i_h, i_w = gt_masks[i].shape
            gt_masks[i] = nn.ConstantPad2d((0,b_w*8-i_w,0,b_h*8-i_h),0)(torch.tensor(gt_masks[i]))
            gt_masks[i] = F.upsample_bilinear(gt_masks[i].float().unsqueeze(0),(b_h,b_w))[0]
        
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs,-1, self.cls_out_channels) for cls_score in cls_scores]
        # need to check images first or dimensions first
        flatten_cls_scores = torch.cat([cls_score for cls_score in flatten_cls_scores],dim=1).reshape(-1, self.cls_out_channels)
        mask_preds = torch.cat([mask_pred for mask_pred in mask_preds],dim=1)

        for i in range(num_imgs):
            ind = torch.nonzero(category_targets[0]).squeeze(-1)
            ins_ind = point_ins[i][ind]
            ins_mask = gt_masks[i][ins_ind].to(mask_preds.device)
            pred_mask = mask_preds[i][ind]
            loss_mask += self.dice_loss(pred_mask, ins_mask)
        loss_mask = loss_mask/num_imgs

        category_targets = torch.cat(category_targets)
        num_pos = (category_targets > 0).sum()
        loss_cls = self.loss_cls(flatten_cls_scores, category_targets, avg_factor=num_pos+num_imgs)
        
        return dict(
               loss_cls=loss_cls,
               loss_mask=loss_mask)       
   
    @force_fp32(apply_to=('cls_scores', 'mask_preds'))
    def get_bboxes(self,cls_scores, mask_preds, img_metas, cfg, rescale=None):
        flatten_cls_scores = torch.cat([cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]).sigmoid()
        _, _, b_h, b_w = mask_preds[0].shape
        for i in range(5):
            mask_preds[i] = F.upsample_bilinear(mask_preds[i],(b_h,b_w))
        mask_preds = torch.cat(mask_preds,dim=1)[0]
        scores, labels = torch.max(flatten_cls_scores,dim=-1)
        nms_pre = cfg.get('nms_pre', -1)
        mask_thr = cfg.get('mask_thr_binary',-1)
        crop_h, crop_w, _ = img_metas[0]['img_shape']
        ori_h, ori_w, _ = img_metas[0]['ori_shape']
       
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            _, topk_inds = scores.topk(nms_pre)
            scores = scores[topk_inds]
            labels = labels[topk_inds]

            mask_preds = mask_preds[topk_inds]
            mask_preds = F.upsample_bilinear(mask_preds.unsqueeze(0), (b_h*8, b_w*8))
            mask_preds = mask_preds[:, :, :crop_h, :crop_w]
            mask_preds = F.sigmoid(F.upsample_bilinear(mask_preds, (ori_h, ori_w)))[0]
            mask_preds = mask_preds > mask_thr

            masks = self.nms(scores, labels, mask_preds, cfg.nms.iou_thr)
            n = len(masks)
            det_bboxes = np.zeros((n, 5))
            det_labels = np.zeros(n).astype(int)
            det_masks = []
            for i in range(n):
                det_bboxes[i, -1] = masks[i][-1]
                det_labels[i] = masks[i][-2]
                det_masks.append(masks[i][0])
            det_masks = np.array(det_masks)
        return det_bboxes, det_labels, det_masks


    def iou_calc(self,mask1,mask2):
        overlap = mask1 & mask2
        union = mask1 | mask2
        iou = float(overlap.sum()+1)/float(union.sum()+1)
        return iou

    def nms(self, scores,labels,masks,iou_threshold=0.5):
        """
        nms function
        :param boxes: list of box
        :param iou_threshold:
        :return:
        """
        return_mask = []
        n = len(labels)
        if n > 0:
            masks_dict = {}
            for i in range(n):
                if labels[i].item() in masks_dict:
                    masks_dict[labels[i].item()].append([masks[i],labels[i],scores[i]])
                else:
                    masks_dict[labels[i].item()] = [[masks[i],labels[i],scores[i]]]
            for masks in masks_dict.values():
                if len(masks) == 1:
                    return_mask.append(masks[0])
                else:
                    while (len(masks)):
                        best_mask = masks.pop(0)
                        return_mask.append(best_mask)
                        j = 0
                        for i in range(len(masks)):
                            i -= j
                            if self.iou_calc(best_mask[0], masks[i][0]) > iou_threshold:
                                masks.pop(i)
                                j += 1
        return return_mask

