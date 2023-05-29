import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from bert.tokenization_bert import BertTokenizer
from lib import lgformer


# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


class args:
    img_path = './demo/people.jpg'
    sentence = ''
    img_size = 480
    swin_type = 'base'
    ck_bert = 'bert-base-uncased'
    window12 = True
    mha = ''
    fusion_drop = 0.0
    checkpoint = './checkpoints/'
    device = torch.device('cuda:0')
    save_path = './demo/seg_res.jpg'


img_transforms = T.Compose([
    T.Resize(args.img_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = lgformer.LGFormer(args=args)
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
model = model.to(args.device)

img = Image.open(args.img_path).convert('RGB')
img_w, img_h = img.size
img_for_vis = np.array(img)

img = img_transforms(img)[None]  # (1, 3, 480, 480)
img = img.to(args.device)

# tokenize input sentence
sentence_tokenized = tokenizer.encode(text=args.sentence, add_special_tokens=True)
sentence_tokenized = sentence_tokenized[:20]
padded_sent_toks = [0] * 20
padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
attention_mask = [0] * 20
attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
padded_sent_toks = torch.tensor(padded_sent_toks)[None]
attention_mask = torch.tensor(attention_mask)[None]
padded_sent_toks = padded_sent_toks.to(args.device)
attention_mask = attention_mask.to(args.device)

# perform segmentation
output = model(img, padded_sent_toks, l_mask=attention_mask[..., None])
output = F.interpolate(output[-1], (img_h, img_w))
output = (output.sigmoid().squeeze(1) > 0.5).cpu().numpy()[0, 0]

# Overlay the mask on the image and save it
output = output.astype(np.uint8)
visualization = overlay_davis(img_for_vis, output)
visualization = Image.fromarray(visualization)
visualization.save(args.save_path)
