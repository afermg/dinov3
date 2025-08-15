import sys

# sys.path.append(REPO_DIR)
from functools import partial

import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms

from dinov3.eval.segmentation.inference import make_inference


def get_img():
    import requests

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "https://phenaid.ardigen.com/static-jumpcpexplorer/images/source_13/CP-CC9-R1-02/A19_5.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


segmentor = torch.hub.load(
    "./",
    "dinov3_vit7b16_ms",
    source="local",
    segmentor_weights="~/projects/dinov3/models/dinov3_vit7b16_ade20k.pth",
    backbone_weights="~/projects/dinov3/models/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
)
segmentor.to("cuda")

img_size = 896
img = get_img()
transform = make_transform(img_size)
# %%
with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img.cuda()
        pred_vit7b = segmentor(batch_img)  # raw predictions
        # actual segmentation map
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)
plt.close()
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map_vit7b[0, 0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")
plt.show()
