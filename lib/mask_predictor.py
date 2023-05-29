import copy

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from .ops.modules import MSDeformAttn
from .position_encoding import PositionEmbeddingSine


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "SyncBN": nn.SyncBatchNorm,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "LN": lambda channels: nn.LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=256, mask_dim=256, nhead=8, dim_ffn=2048, dropout=0.1, num_queries=1):
        super().__init__()
        self.layers = _get_clones(
            TransformerDecoderLayer(d_model=d_model, mask_dim=mask_dim, nhead=nhead, dim_feedforward=dim_ffn, dropout=dropout),
            num_layers
        )
        self.num_layers = num_layers

        self.pe_layer = PositionEmbeddingSine(mask_dim // 2, normalize=True)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, mask_dim)

    def forward(self, inputs):
        '''
            txt: (B, L, D)
            ms_feats: List(3, )
        '''
        txt, ms_feats = inputs
        B, _, _ = txt.size()
        _, C, _, _ = ms_feats[0].size()
        txt = txt.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

        # position embedding
        pos = []
        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(ms_feats[i], None).flatten(2).permute(2, 0, 1))

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        outputs = []
        output = txt
        outputs.append(output.permute(1, 0, 2))
        for i in range(self.num_layers):
            idx = i % len(ms_feats)
            vis = ms_feats[idx].reshape(B, C, -1) + self.level_embed.weight[idx][None, :, None]
            vis = vis.permute(2, 0, 1)
            output = self.layers[i](output, vis, key_pos=pos[idx], query_pos=query_embed)
            outputs.append(output.permute(1, 0, 2))     # (L, B, D) -> (B, L, D)

        return outputs


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, mask_dim=256, nhead=8, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Attention Layer
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=mask_dim, vdim=mask_dim)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ca = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, key_pos, query_pos, key_mask=None):
        # Cross Attention
        query2 = self.cross_attn(query=self.with_pos_embed(query, query_pos),
                                 key=self.with_pos_embed(key, key_pos),
                                 value=key,
                                 key_padding_mask=key_mask)[0]
        query = query + self.dropout_ca(query2)
        query = self.norm_ca(query)

        # FFN
        query2 = self.ffn(query)
        query = query + self.dropout_ffn(query2)
        query = self.norm_ffn(query)
        return query


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super(MSDeformAttnTransformerEncoderLayer, self).__init__()

        # Self Attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # Self Attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, num_layers, text_len=20, text_dim=768):
        super(MSDeformAttnTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.cross_attn_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        
        self.key_embed = nn.Embedding(text_len, text_dim)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # spatial_shapes: [[h0, w0], [h1, w1], ...]
        # valid_ratios: (bs, num_levels, 2)
        reference_points_list = []
        for lvl, (h_, w_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h_ - 0.5, h_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, w_ - 0.5, w_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, txt=None, txt_mask=None):
        output = src
        key_pos = self.key_embed.weight.unsqueeze(1).repeat(1, txt.shape[0], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer, cross_attn_layer in zip(self.layers, self.cross_attn_layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            output = cross_attn_layer(output.permute(1, 0, 2).float(), txt.permute(1, 0, 2).float(), key_pos.float(), pos.permute(1, 0, 2).float(), key_mask=~(txt_mask.squeeze(-1).float().bool()))
            output = output.permute(1, 0, 2).float()

        return output


# encoder only
class MSDeformAttnTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, activation="relu",
                 num_feature_levels=4, enc_n_points=4):
        super(MSDeformAttnTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        cross_attn_layer = TransformerDecoderLayer(d_model=d_model, mask_dim=768, nhead=8, dim_feedforward=2048, dropout=0.0)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, cross_attn_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, txt, txt_mask):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)

            mask = mask.flatten(1)
            mask_flatten.append(mask)

            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]), 0)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, txt=txt, txt_mask=txt_mask)
        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, input_shape, transformer_dropout, transformer_nheads, transformer_dim_feedforward,
                 transformer_enc_layers, conv_dim, mask_dim, norm, transformer_in_features, common_stride):
        super(MSDeformAttnPixelDecoder, self).__init__()

        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][0])
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v[0] for k, v in input_shape]
        self.feature_channels = [v[1] for k, v in input_shape]

        # input shape of transformer decoder which use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1][0])
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v[1] for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v[0] for k, v in transformer_input_shape]

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (s4 -> s2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim)
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim)
            )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformer(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)
        nn.init.kaiming_uniform_(self.mask_features.weight, a=1)
        nn.init.constant_(self.mask_features.bias, 0)

        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = _get_norm(norm, conv_dim)
            output_norm = _get_norm(norm, conv_dim)

            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias),
                lateral_norm
            )
            nn.init.kaiming_uniform_(lateral_conv[0].weight, a=1)
            if use_bias:
                nn.init.constant_(lateral_conv[0].bias, 0)

            output_conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                output_norm,
                nn.ReLU(inplace=True)
            )
            nn.init.kaiming_uniform_(output_conv[0].weight, a=1)
            if use_bias:
                nn.init.constant_(output_conv[0].bias, 0)

            self.add_module(f"adapter_{idx + 1}", lateral_conv)
            self.add_module(f"layer_{idx + 1}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # place convs into top-down order
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @autocast(enabled=False)
    def forward(self, features, txt, txt_mask):
        # txt: (B, L, D), txt_mask: (B, L, 1)
        srcs = []
        pos = []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos, txt, txt_mask)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)   # multi-scale features

        out = []
        multi_scale_features = []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        num_cur_levels = 0
        for o in out:
            if num_cur_levels < self.transformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), multi_scale_features


class MSDeformDynamicConvDecoding(nn.Module):
    def __init__(self, mask_dim=256, txt_dim=768, swin_embed_dim=128, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        input_shape = {}
        for i in range(4):
            input_shape[f"s{i + 1}"] = (2 ** (i + 2), swin_embed_dim * (2 ** i))

        self.pix_dec = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=mask_dim,
            norm="GN",
            transformer_in_features=["s2", "s3", "s4"],
            common_stride=4
        )

        # generate dynamic convolution filter
        self.txt_proj = nn.Linear(txt_dim, txt_dim)
        self.dynamic_conv_gen_body = TransformerDecoder(num_layers=6, d_model=txt_dim, mask_dim=mask_dim, nhead=8, dim_ffn=2048, dropout=0.0)
        self.dynamic_conv_gen_head = nn.Sequential(
            nn.Linear(txt_dim, txt_dim),
            nn.ReLU(True),
            nn.Linear(txt_dim, txt_dim),
            nn.ReLU(True),
            nn.Linear(txt_dim, 1 * mask_dim * kernel_size * kernel_size + 1)
        )

    def forward(self, image_encodings, text_encodings, text_mask):
        text_encodings = text_encodings.permute(0, 2, 1)
        input_features = {}
        for i, feats in enumerate(image_encodings, start=1):
            input_features[f's{i}'] = feats

        # pixel decoding
        x, ms_feats = self.pix_dec(input_features, text_encodings, text_mask)

        # pool text embeddings into sentence embeddings
        text_mask = text_mask.expand(text_encodings.size()).float()
        text_embed = torch.sum(text_encodings * text_mask, dim=1, keepdim=True)
        text_embed = text_embed / torch.clamp(text_mask.sum(dim=1, keepdim=True), min=1e-6)
        text_embed = self.txt_proj(text_embed)

        dynamic_convs = self.dynamic_conv_gen_body((text_embed, ms_feats))

        outs = []
        for lno in range(len(dynamic_convs)):
            B, Q, _ = dynamic_convs[lno].shape
            dynamic_conv = self.dynamic_conv_gen_head(dynamic_convs[lno])
            weight, bias = dynamic_conv[:, :, :-1], dynamic_conv[:, :, -1]
            out = torch.einsum("bchw,bqc->bqhw", x, weight)
            out = out + bias.reshape([B, Q, 1, 1])
            outs.append(out)

        return outs
