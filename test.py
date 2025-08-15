import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

weights_path = "~/Downloads/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
# dinov3_convnext_base = torch.hub.load("./", "dinov3_convnext_base", source="local", weights=weights_path)
model = torch.hub.load("./", "dinov3_convnext_base", source="local", weights=weights_path)
pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(
#     weights_path,
#     device_map="auto",
# )

# inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# pooled_output = outputs.pooler_output
# print("Pooled output shape:", pooled_output.shape)
