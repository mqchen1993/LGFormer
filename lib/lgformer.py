import torch.nn as nn
import torch.nn.functional as F

from bert.modeling_bert import BertModel
from .backbone import MultiModalSwinTransformer
from .mask_predictor import MSDeformDynamicConvDecoding


class LGFormer(nn.Module):
    def __init__(self, args):
        super(LGFormer, self).__init__()
        # Swin transformer encoder
        swin_cfgs = {
            'tiny': (96, [2, 2, 6, 2], [3, 6, 12, 24]),
            'small': (96, [2, 2, 18, 2], [3, 6, 12, 24]),
            'base': (128, [2, 2, 18, 2], [4, 8, 16, 32]),
            'large': (192, [2, 2, 18, 2], [6, 12, 24, 48])
        }
        embed_dim, depths, num_heads = swin_cfgs[args.swin_type]

        if 'window12' in args.pretrained_swin_weights or args.window12:
            print('Window size 12!')
            window_size = 12
        else:
            window_size = 7
        
        mha = [int(a) for a in args.mha.split('-')] if args.mha else [1, 1, 1, 1]

        self.backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                  window_size=window_size,
                                                  ape=False, drop_path_rate=0.3, patch_norm=True,
                                                  use_checkpoint=False, num_heads_fusion=mha,
                                                  fusion_drop=args.fusion_drop
                                                  )
        if args.pretrained_swin_weights:
            print('Initializing Multi-modal Swin Transformer weights from ' + args.pretrained_swin_weights)
            self.backbone.init_weights(pretrained=args.pretrained_swin_weights)
        else:
            print('Randomly initialize Multi-modal Swin Transformer weights.')
            self.backbone.init_weights()

        # MSDeform Decoder
        self.classifier = MSDeformDynamicConvDecoding(mask_dim=256, txt_dim=768, swin_embed_dim=embed_dim, kernel_size=1)

        # text encoder
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, image, text, text_mask):
        # text encoding
        text_encodings = self.text_encoder(text, attention_mask=text_mask)[0]   # (B, L, D)
        text_encodings = text_encodings.permute(0, 2, 1)    # (B, L, D) -> (B, D, L)
        text_mask = text_mask[..., None]    # (B, L, 1)
        
        # image encoding
        image_encodings, text_encodings = self.backbone(image, text_encodings, text_mask)

        # mask decoding
        masks = self.classifier(image_encodings, text_encodings, text_mask)
        masks = [F.interpolate(m, size=image.shape[-2:], mode='bilinear', align_corners=False) for m in masks]

        return masks
