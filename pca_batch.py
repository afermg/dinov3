import urllib

# sys.path.append(REPO_DIR)
from functools import partial

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from matplotlib import colormaps
from PIL import Image


from skimage.filters import threshold_otsu
from scipy import signal

from dinov3.eval.segmentation.inference import make_inference

PATCH_SIZE = 16
IMAGE_SIZE = (1728, 1728)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def get_img():
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "https://phenaid.ardigen.com/static-jumpcpexplorer/images/source_13/CP-CC9-R1-02/A19_5.jpg"
    url = "https://images.proteinatlas.org/12047/95_E8_1_blue_red_green.jpg"
    url1 = "https://images.proteinatlas.org/38742/471_D3_1_blue_red_green.jpg"
    url2 = "https://images.proteinatlas.org/50524/711_D6_1_blue_red_green.jpg"
    image1 = load_image_from_url(url1)
    image2 = load_image_from_url(url2)
    return image1, image2


def resize_transform(
    mask_image: Image,
    image_size: tuple = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    IMAGE_SIZE = mask_image.size

    print(f"Original image size: {w}x{h}")
    h_patches = int(image_size[1] / patch_size)
    w_patches = int((w * image_size[0]) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))



# base_path = "~/projects/dinov3/models/"
base_path = "/home/jfredinh/projects/DINOv3/dinov3/model_download/"

model = torch.hub.load(
    "./",
    "dinov3_vit7b16",
    source="local",
    weights=f"{base_path}/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
)


model.to("cuda")


image1, image2 = get_img()
image_resized1 = resize_transform(image1)    
image_resized_norm1 = TF.normalize(image_resized1, mean=IMAGENET_MEAN, std=IMAGENET_STD)

image_resized2 = resize_transform(image2)    
image_resized_norm2 = TF.normalize(image_resized2, mean=IMAGENET_MEAN, std=IMAGENET_STD)

#image_resized_norm2 = torch.flip(image_resized_norm1, [0, 1])

# %%
n_layers = 40

with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        feats1 = model.get_intermediate_layers(
            image_resized_norm1.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True
        )
        x1 = feats1[-1].squeeze().detach().cpu()
        dim = x1.shape[0]
        x1 = x1.view(dim, -1).permute(1, 0)

        feats2 = model.get_intermediate_layers(
            image_resized_norm2.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True
        )
        x2 = feats2[-1].squeeze().detach().cpu()
        dim = x2.shape[0]
        x2 = x2.view(dim, -1).permute(1, 0)


h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized1.shape[1:]]


# Adapted version of the PCA code from the DINOv3 repository
# The original code uses a foreground background segmentation approach
# to select patches, but here we instead use the first PCA component to seperate foreground from background.

from sklearn.decomposition import PCA

pca = PCA(n_components=4, whiten=True)
pca.fit(x1)
# apply the PCA, and then reshape

pca_sep = pca.transform(x1.numpy())[:,0] 

fg_score = pca_sep.reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

threshold = threshold_otsu(pca_sep,nbins=100) 
print(f"Threshold for foreground selection: {threshold}")
foreground_selection = fg_score_mf.view(-1) > threshold
fg_patches = x1[foreground_selection]




pca = PCA(n_components=4, whiten=True)
pca.fit(x2)
# apply the PCA, and then reshape

pca_sep2 = pca.transform(x2.numpy())[:,0]

fg_score2 = pca_sep2.reshape(h_patches, w_patches)
fg_score_mf2 = torch.from_numpy(signal.medfilt2d(fg_score2, kernel_size=3))

threshold2 = threshold_otsu(pca_sep2,nbins=100) 
print(f"Threshold for foreground selection: {threshold2}")
foreground_selection2 = fg_score_mf2.view(-1) > threshold2
fg_patches2 = x2[foreground_selection2]




plt.figure()
# Create the histogram
plt.hist(pca_sep, bins=100, alpha=0.7)

# Add labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Foreground Scores")

# Display the plot
plt.savefig("HPA_pca_histogram.png", dpi=300, bbox_inches='tight')
plt.close()


# Once foreground patches are selected, we can apply PCA to these patches

pca = PCA(n_components=3, whiten=True)
pca.fit(torch.concat([fg_patches, fg_patches2]))




# apply the PCA, and then reshape
projected_image = torch.from_numpy(pca.transform(x1.numpy())).view(h_patches, w_patches, 3)
# multiply by 2.0 and pass through a sigmoid to get vibrant colors
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
#projected_image = torch.nn.functional.sigmoid(projected_image.mul(1.0)).permute(2, 0, 1)
# mask the background using the fg_score_mf
projected_image *= fg_score_mf.unsqueeze(0) > threshold


# apply the PCA, and then reshape
projected_image2 = torch.from_numpy(pca.transform(x2.numpy())).view(h_patches, w_patches, 3)
# multiply by 2.0 and pass through a sigmoid to get vibrant colors
projected_image2 = torch.nn.functional.sigmoid(projected_image2.mul(2.0)).permute(2, 0, 1)
#projected_image = torch.nn.functional.sigmoid(projected_image.mul(1.0)).permute(2, 0, 1)
# mask the background using the fg_score_mf
projected_image2 *= fg_score_mf2.unsqueeze(0) > threshold2




# enjoy
plt.figure(dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis("off")
plt.show()
plt.savefig("HPA_pca_projection.png", dpi=300, bbox_inches='tight')
plt.close()




plt.figure(figsize=(12, 6))
plt.subplot(221)
plt.imshow(image1)
plt.axis("off")
plt.subplot(222)
plt.imshow(image2)
plt.axis("off")
plt.subplot(223)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis("off")
plt.subplot(224)
plt.imshow(projected_image2.permute(1, 2, 0))
plt.axis("off")
plt.savefig("HPA_image_combined_flipped.png", dpi=300, bbox_inches='tight')
plt.show()
