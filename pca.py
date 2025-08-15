import urllib

# sys.path.append(REPO_DIR)
from functools import partial

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from matplotlib import colormaps
from PIL import Image

from dinov3.eval.segmentation.inference import make_inference

PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def get_img():
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "https://phenaid.ardigen.com/static-jumpcpexplorer/images/source_13/CP-CC9-R1-02/A19_5.jpg"
    image = load_image_from_url(url)
    return image


def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


image = get_img()
image_resized = resize_transform(image)
image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

model = torch.hub.load(
    "./",
    "dinov3_vit7b16_ms",
    source="local",
    segmentor_weights="~/projects/dinov3/models/dinov3_vit7b16_ade20k.pth",
    backbone_weights="~/projects/dinov3/models/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    # pretrained=False,
)
# state_dict = torch.load(
#     "/home/amunozgo/projects/dinov3/models/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", map_location="cpu"
# )
# model.load_state_dict(state_dict)
model.to("cuda")

# %%
n_layers = 40

with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        feats = model.get_intermediate_layers(
            image_resized_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True
        )
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]

fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

plt.rcParams.update({
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.labelsize": 5,
    "axes.titlesize": 4,
})

plt.figure(figsize=(4, 2), dpi=300)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis("off")
plt.title(f"Image, Size {image.size}")
plt.subplot(1, 2, 2)
plt.imshow(fg_score_mf)
plt.title(f"Foreground Score, Size {tuple(fg_score_mf.shape)}")
plt.colorbar()
plt.axis("off")
plt.show()

from sklearn.decomposition import PCA

foreground_selection = fg_score_mf.view(-1) > 0.5
fg_patches = x[foreground_selection]
pca = PCA(n_components=3, whiten=True)
pca.fit(fg_patches)
# apply the PCA, and then reshape
projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)

# multiply by 2.0 and pass through a sigmoid to get vibrant colors
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

# mask the background using the fg_score_mf
projected_image *= fg_score_mf.unsqueeze(0) > 0.5

# enjoy
plt.figure(dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis("off")
plt.show()
