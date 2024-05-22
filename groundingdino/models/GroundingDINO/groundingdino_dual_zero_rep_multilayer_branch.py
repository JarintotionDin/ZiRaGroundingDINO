# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List, Union

import torch
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer_for_adapter import build_transformer
# from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss, recover_to_cls_logits, ContrastiveEmbedwithLinear

from .criterion import build_criterion

from ...util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
import random


zero_value = 1e-8

class ZeroGroupNorm(nn.GroupNorm):
    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.constant_(self.weight, val=zero_value)
            if self.bias is not None:
                nn.init.constant_(self.bias, val=zero_value)

class RepZeroConv2dGN(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding = 0,
                 dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, zero_value=zero_value) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.scaling = nn.parameter.Parameter(torch.ones(1) * 1.0)
        nn.init.constant_(self.weight, val=zero_value)
        if self.bias is not None:
            nn.init.constant_(self.bias, val=zero_value)
        
        self.freeze_conv = nn.Conv2d(in_channels, out_channels,
                                     kernel_size, stride, padding,
                                     dilation, groups, bias,
                                     padding_mode, device, dtype)
        self.freeze_gn = ZeroGroupNorm(32, out_channels)
        nn.init.constant_(self.freeze_conv.weight, val=0.0)
        if self.bias is not None:
            nn.init.constant_(self.freeze_conv.bias, val=0.0)
        
        self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            branch_output = super().forward(input) * self.scaling
            output = self.freeze_gn(branch_output + self.freeze_conv(input))
            return output, \
                self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
                    self.zero_inter_loss(output, torch.zeros_like(output))
            # return output, \
            #     self.zero_inter_loss(branch_output, torch.zeros_like(branch_output))
        else:
            return self.freeze_conv(input), torch.zeros(1).to(input)
    
    # def forward(self, input: Tensor) -> Tensor:
    #     branch_output = super().forward(input)
    #     output = branch_output + self.freeze_conv(input)
    #     return output * self.scaling, self.zero_inter_loss(output * self.scaling, torch.zeros_like(output))

    def __rep__(self):
        self.freeze_conv.weight.data = self.weight.data * self.scaling + self.freeze_conv.weight.data
        self.freeze_conv.bias.data = self.bias.data  * self.scaling + self.freeze_conv.bias.data
        self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) * 1.0)
        nn.init.constant_(self.weight, val=zero_value)
        if self.bias is not None:
            nn.init.constant_(self.bias, val=zero_value)


class RepZeroLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scaling = nn.parameter.Parameter(torch.ones(1) * 1.0)
        nn.init.constant_(self.weight, val=zero_value)
        self.freeze_linear = nn.Linear(in_features, out_features, bias, device, dtype)
        nn.init.constant_(self.freeze_linear.weight, val=0.0)
        if self.bias is not None:
            nn.init.constant_(self.freeze_linear.bias, val=0.0) 
        
        self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            branch_output = self.scaling * super().forward(input)
            output = branch_output + self.freeze_linear(input)
            return output, \
                self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
                    self.zero_inter_loss(output, torch.zeros_like(output))
            # return output, \
            #     self.zero_inter_loss(branch_output, torch.zeros_like(branch_output))
        else:
            return self.freeze_linear(input), torch.zeros(1).to(input)

    def __rep__(self):
        self.freeze_linear.weight.data = self.weight.data  * self.scaling + self.freeze_linear.weight.data
        self.freeze_linear.bias.data = self.bias.data  * self.scaling + self.freeze_linear.bias.data
        self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) * 1.0)
        nn.init.constant_(self.weight, val=zero_value)
        if self.bias is not None:
            nn.init.constant_(self.bias, val=zero_value)


class RepZeroTransformerLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 nhead=8,
                 down_dim=2048,
                 ffn_drop=0,
                 activation=nn.ReLU(inplace=True),
                 output_dim=None,
                 **kwargs,) -> None: 
        super().__init__()
        if output_dim is None:
            output_dim = embed_dim
        self.freeze_self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=ffn_drop)
        self.freeze_linear1 = nn.Linear(embed_dim, down_dim)
        self.dropout = nn.Dropout(ffn_drop)
        self.freeze_linear2 = nn.Linear(down_dim, output_dim)
        self.freeze_norm1 = nn.LayerNorm(embed_dim)
        self.freeze_norm2 = nn.LayerNorm(output_dim)
        self.dropout1 = nn.Dropout(ffn_drop)
        self.dropout2 = nn.Dropout(ffn_drop)
        self.nhead = nhead
        self.activation = activation
  
        nn.init.zeros_(self.freeze_linear2.weight)
        nn.init.zeros_(self.freeze_linear2.bias)
        
        self.free_linear1 = nn.Linear(embed_dim, down_dim)
        self.free_linear2 = nn.Linear(down_dim, output_dim)
        nn.init.constant_(self.free_linear1.weight, val=zero_value)
        nn.init.constant_(self.free_linear1.bias, val=zero_value)
        nn.init.constant_(self.free_linear2.weight, val=zero_value)
        nn.init.constant_(self.free_linear2.bias, val=zero_value)

        self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')

    def __rep__(self):
        self.freeze_linear1.weight.data = self.free_linear1.weight.data  + self.freeze_linear1.weight.data
        self.freeze_linear2.weight.data = self.free_linear2.weight.data  + self.freeze_linear2.weight.data
        self.freeze_linear1.bias.data = self.free_linear1.bias.data  + self.freeze_linear1.bias.data
        self.freeze_linear2.bias.data = self.free_linear2.bias.data  + self.freeze_linear2.bias.data
        nn.init.constant_(self.free_linear1.weight, val=zero_value)
        nn.init.constant_(self.free_linear1.bias, val=zero_value)
        nn.init.constant_(self.free_linear2.weight, val=zero_value)
        nn.init.constant_(self.free_linear2.bias, val=zero_value)
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        pos=None,
    ):
        if self.training:
            q = k = self.with_pos_embed(src, pos)
            src0 = self.freeze_self_attn(q, k, value=src)[0]
            src = src + self.dropout1(src0)
            src = self.freeze_norm1(src)
            branch_out1 = self.free_linear1(src)
            src1 = self.freeze_linear1(src) + branch_out1
            src1 = self.dropout(self.activation(src1))
            branch_out2 = self.free_linear2(src1)
            src2 = self.freeze_linear2(src1) + branch_out2
            src = self.dropout2(src2)
            src = self.freeze_norm2(src)
            return src, self.zero_inter_loss(branch_out1, torch.zeros_like(branch_out1)) + \
                        self.zero_inter_loss(branch_out2, torch.zeros_like(branch_out2)) + \
                        self.zero_inter_loss(src, torch.zeros_like(src))
        else:
            q = k = self.with_pos_embed(src, pos)
            src0 = self.freeze_self_attn(q, k, value=src)[0]
            src = src + self.dropout1(src0)
            src = self.freeze_norm1(src)
            src1 = self.dropout(self.activation(self.freeze_linear1(src)))
            src2 = self.freeze_linear2(src1)
            src = self.dropout2(src2)
            src = self.freeze_norm2(src)
            zero_loss = torch.zeros(1).to(src)
            return src, zero_loss


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        criterion=None,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [123.675, 116.280, 103.530],
        device="cuda",
        select_box_nums_for_evaluation=200,
        # pat
        freeze_all=False,
        loss_adapter_weight=0.1,
        use_cet=False,
        use_prompt_memory=False,
        num_select_prompt=200,
        use_zero_inter_loss=True,
        cet_middle_dim=64,
        use_add_names=False,
        use_bert_tuning=False,
        num_experts=1,
        topk=1,
        use_cls_linear=False,
        use_prompt_tuning=False,
        use_prompt_memory_output=True, # use prompt memory to replace the output of language adapter 
        cet_type="Adapter",
        zero_loss_type="L1",
        use_project_tuning=False,
        use_project_adapter=True,
        use_zero_inter_loss_for_conv=True,
        use_learned_names=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)
        
        # bert text feature maper
        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        self.rep_language_adapter = RepZeroLinear(in_features=self.bert.config.hidden_size,
                                                  out_features=self.hidden_dim)
        # self.rep_language_adapter = RepZeroTransformerLayer(embed_dim=self.bert.config.hidden_size,
        #                                                     output_dim=self.hidden_dim)
        
        # learned classes
        self.learned_classes = []
        
        # for language adapter tuning
        self.use_cet = use_cet
        self.use_prompt_memory = use_prompt_memory
        self.use_zero_inter_loss = use_zero_inter_loss
        self.prompt_memory_pool = nn.ParameterDict()
        self.num_select_prompt = num_select_prompt
        self.use_prompt_tuning = use_prompt_tuning
        self.use_learned_names = use_learned_names
        self.use_prompt_memory_output = use_prompt_memory_output

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        # projection adapter, keep the same as feature projection layers
        self.use_project_adapter = use_project_adapter
        self.use_zero_inter_loss_for_conv = use_zero_inter_loss_for_conv
        if use_project_adapter:
            if num_feature_levels > 1:
                num_backbone_outs = len(backbone.num_channels)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    input_proj_list.append(RepZeroConv2dGN(in_channels, hidden_dim, kernel_size=1))
                for _ in range(num_feature_levels - num_backbone_outs):
                    input_proj_list.append(RepZeroConv2dGN(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1))
                    in_channels = hidden_dim
                self.input_proj_conv_adapter = nn.ModuleList(input_proj_list)
            else:
                assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
                self.input_proj_conv_adapter = nn.ModuleList([RepZeroConv2dGN(backbone.num_channels[-1], hidden_dim, kernel_size=1)])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        if use_cls_linear:
            _class_embed = ContrastiveEmbedwithLinear(max_text_len=self.max_text_len,
                                                      hidden_dim=self.hidden_dim)
        else:
            _class_embed = ContrastiveEmbed(max_text_len=self.max_text_len)

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
            
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
        
        # critetion
        self.criterion = criterion
        
        #
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        
        #  device
        self.device = device
        
        # init param
        self._reset_parameters()
        
        #  add names
        self.use_add_names = use_add_names
        
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        
        self.loss_adapter_weight = loss_adapter_weight
        
        # freeze
        self.freeze_all = freeze_all
        
        # bert-tuning
        self.use_bert_tuning = use_bert_tuning
        
        # class linear
        self.use_cls_linear = use_cls_linear
        
        # input project
        self.use_project_tuning = use_project_tuning

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, batched_inputs, **kw):
        # process images
        images = self.preprocess_image(batched_inputs)
        assert isinstance(images, ImageList)
        samples = nested_tensor_from_tensor_list(images)

        # prepare captions
        captions = [x["captions"] for x in batched_inputs]
        names_list = [x["captions"][:-1].split(".") for x in batched_inputs]

        if (self.use_add_names and (not self.training)) or (self.use_learned_names and self.training):
            non_overlap_classes = [class_name for class_name in \
                self.learned_classes if (class_name not in names_list[0])]
            if self.training and (len(non_overlap_classes) >= self.num_select_prompt):
                    non_overlap_classes = random.sample(non_overlap_classes, self.num_select_prompt)    
            for idx, (caption, names) in enumerate(zip(captions, names_list)):
                names_list[idx] = names + non_overlap_classes
                captions[idx] = caption + ".".join(non_overlap_classes)
                if not captions[idx].endswith("."): captions[idx] = captions[idx] + "."

        tokenized = self.tokenizer(captions, padding="longest",
                                   return_tensors="pt").to(samples.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        # prepare targets
        targets = None
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, cate_to_token_mask_list, names_list)

        # encoder texts
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
        
        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        rep_language_out, loss_language_adapter = self.rep_language_adapter(bert_output["last_hidden_state"])
        encoded_text = rep_language_out + encoded_text
        
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        loss_conv_adapter = None
        
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            if not self.use_project_adapter:
                srcs.append(self.input_proj[l](src))
            else:
                conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](src)
                srcs.append(self.input_proj[l](src) + conv_adapter_output)
                if loss_conv_adapter is None:
                    loss_conv_adapter = zero_loss
                else:
                    loss_conv_adapter = loss_conv_adapter + zero_loss
            masks.append(mask)
            assert mask is not None
            
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    if not self.use_project_adapter:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](features[-1].tensors)
                        src = self.input_proj[l](features[-1].tensors) + conv_adapter_output
                        if loss_conv_adapter is None:
                            loss_conv_adapter = zero_loss
                        else:
                            loss_conv_adapter = loss_conv_adapter + zero_loss
                else:
                    if not self.use_project_adapter:
                        src = self.input_proj[l](srcs[-1])
                    else:
                        conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](srcs[-1])
                        src = self.input_proj[l](srcs[-1]) + conv_adapter_output
                        if loss_conv_adapter is None:
                            loss_conv_adapter = zero_loss
                        else:
                            loss_conv_adapter = loss_conv_adapter + zero_loss
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal, adapter_loss = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                recover_to_cls_logits(layer_cls_embed(layer_hs, text_dict), \
                    cate_to_token_mask_list, for_fill=-100.0)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ])
        
        # print(outputs_class.shape)
        out = {"pred_logits": outputs_class[-1],
               "pred_boxes": outputs_coord_list[-1],
               "cate_to_token_mask_list": cate_to_token_mask_list}


        if self.training:
            # for intermediate outputs
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

            # for encoder output
            if hs_enc is not None:
                # prepare intermediate outputs
                interm_coord = ref_enc[-1]
                interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
                interm_class = recover_to_cls_logits(interm_class, cate_to_token_mask_list, for_fill=-100.0)
                out["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}
                
            # return loss
            assert targets is not None
            assert self.criterion is not None
            loss_dict = self.criterion(out, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]

            if self.use_project_adapter and self.use_zero_inter_loss_for_conv:
                loss_dict["loss_conv_adapter"] = loss_conv_adapter * self.loss_adapter_weight
            if self.use_cet and self.use_zero_inter_loss:
                loss_dict["loss_language_adapter"] = loss_language_adapter * self.loss_adapter_weight
            return loss_dict
        else:
            box_cls = out["pred_logits"]
            box_pred = out["pred_boxes"]
            results = self.dt_inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_targets(self, targets, cate_to_token_mask_list, names_list):
        new_targets = []
        for targets_per_image, cate_to_token_mask, names in \
            zip(targets, cate_to_token_mask_list, names_list):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            # gt_class_names = targets_per_image.gt_class_names
            # gt_classes =  torch.as_tensor([names.index(name) for name in gt_class_names],
            #                                        dtype=torch.long, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def dt_inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def normalizer(self, x):
        pixel_mean = torch.Tensor(self.pixel_mean).to(x.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.pixel_std).to(x.device).view(3, 1, 1)
        return (x - pixel_mean) / pixel_std

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def unfreeze_module_(self, pat_names):
        for module_name, param in self.named_parameters():
            for pat_name in pat_names:
                if pat_name in module_name:
                    param.requires_grad = True
                    print("unfreeze:", module_name)
                    break
    
    def freeze_model_(self):
        for module_name, param in self.named_parameters():
            if "freeze" in module_name:
                param.requires_grad = False
                print("freeze:", module_name)
        
    
    def load_state_dict(self, state_dict, strict=True):
        res = super().load_state_dict(state_dict=state_dict, strict=strict)
        return res
    
    def before_train(self):
        if self.freeze_all:
            for param in self.parameters():
                param.requires_grad = False
        if self.use_bert_tuning:
            self.unfreeze_module_(["bert", "feat_map"])
        if self.use_cls_linear:
            self.unfreeze_module_(["class_embed", "bbox_embed"])
        if self.use_prompt_tuning:
            self.unfreeze_module_(["prompt_memory_pool"])
        if self.use_project_tuning:
            self.unfreeze_module_(["input_proj"])
        self.unfreeze_module_(["adapter"])
    
        # self.freeze_model_()
    
    def after_train(self):
        # for rep
        for module in self.modules():
            if hasattr(module, '__rep__'):
                module.__rep__() 
        

    @torch.no_grad()
    def inference(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
        # names_list = [caption[:-1].split(".") for caption in captions]
        names_list = captions[0][:-1].split(".")
        names_list = [names_list]

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # captions = [x["captions"] for x in batched_inputs]
        # names_list = [x["captions"][:-1].split(".") for x in batched_inputs]

        if (self.use_add_names and (not self.training)) or (self.use_learned_names and self.training):
            non_overlap_classes = [class_name for class_name in \
                self.learned_classes if (class_name not in names_list[0])]
            if self.training and (len(non_overlap_classes) >= self.num_select_prompt):
                    non_overlap_classes = random.sample(non_overlap_classes, self.num_select_prompt)    
            for idx, (caption, names) in enumerate(zip(captions, names_list)):
                names_list[idx] = names + non_overlap_classes
                captions[idx] = caption + ".".join(non_overlap_classes)
                if not captions[idx].endswith("."): captions[idx] = captions[idx] + "."

        tokenized = self.tokenizer(captions, padding="longest",
                                   return_tensors="pt").to(samples.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        # prepare targets
        targets = None
        # if self.training:
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        #     targets = self.prepare_targets(gt_instances, cate_to_token_mask_list, names_list)

        # encoder texts
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
        
        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        
        if self.use_cet and (not self.use_prompt_memory_output):
            adapter_output, adapter_loss_text = self.cet_adapter(bert_output["last_hidden_state"])
            if self.use_zero_inter_loss:
                adapter_loss_text = adapter_loss_text + \
                    self.adapter_loss(adapter_output, torch.zeros_like(adapter_output))
                    
            encoded_text = encoded_text + adapter_output
        
        if self.use_prompt_tuning or self.use_prompt_memory_output:
            encoded_text_target = encoded_text.clone()
            for bid, cate_to_token_mask in enumerate(cate_to_token_mask_list):
                for cate_cid in range(len(cate_to_token_mask)):
                    class_name = names_list[bid][cate_cid]
                    class_name = "-{}-".format(class_name)
                    if class_name in self.prompt_memory_pool.keys():
                        prompt_memory = self.prompt_memory_pool[class_name]
                        encoded_text_target[bid, :cate_to_token_mask.shape[1], 
                                            :][cate_to_token_mask[cate_cid]] = prompt_memory
            encoded_text = encoded_text_target
        
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal, adapter_loss = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1], "cate_to_token_mask_list": cate_to_token_mask_list}
        return out

@MODULE_BUILD_FUNCS.registe_with_name(module_name="dualzerorepmultilayerbranchgroundingdino")
def build_dual_zero_rep_multi_layer_branch_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    criterion = build_criterion(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present
    
    if hasattr(args, "use_project_tuning"):
        use_project_tuning = args.use_project_tuning
    else:
        use_project_tuning = False
    
    if hasattr(args, "use_learned_names"):
        use_learned_names = args.use_learned_names
    else:
        use_learned_names = False

    if hasattr(args, "use_project_adapter"):
        use_project_adapter = args.use_project_adapter
    else:
        use_project_adapter = False

    if hasattr(args, "use_zero_inter_loss_for_conv"):
        use_zero_inter_loss_for_conv = args.use_zero_inter_loss_for_conv
    else:
        use_zero_inter_loss_for_conv = False

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        criterion=criterion,
        freeze_all=args.freeze_all,
        select_box_nums_for_evaluation=args.select_box_nums_for_evaluation,
        use_cet=args.use_cet,
        use_prompt_memory=args.use_prompt_memory,
        use_zero_inter_loss=args.use_zero_inter_loss,
        loss_adapter_weight=args.loss_adapter_weight,
        num_experts=args.num_experts,
        topk=args.num_topk_experts,
        cet_middle_dim=args.cet_middle_dim,
        use_bert_tuning=args.use_bert_tuning,
        use_cls_linear=args.use_cls_linear,
        use_prompt_tuning=args.use_prompt_tuning,
        use_prompt_memory_output=args.use_prompt_memory_output,
        cet_type=args.cet_type,
        use_project_tuning=use_project_tuning,
        use_learned_names=use_learned_names,
        use_project_adapter=use_project_adapter,
        use_zero_inter_loss_for_conv=use_zero_inter_loss_for_conv,
    )

    return model
