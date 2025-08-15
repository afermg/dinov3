import torch
from torchvision import transforms
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = load_image(url)

weights_path = "~/Downloads/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
# dinov3_convnext_base = torch.hub.load("./", "dinov3_convnext_base", source="local", weights=weights_path)
model = torch.hub.load("./", "dinov3_convnext_tiny", source="local", weights=weights_path)
pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


def make_transform(resize_size: int = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


# %%
transform = make_transform()
with torch.inference_mode():
    batch_img = transform(img)[None]
    result = model(batch_img)
