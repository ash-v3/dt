import torch
from torch import nn, return_types
from torch.nn import functional as F
import torchvision.transforms as T
import transformers
from PIL import Image


class StateProc(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        # model_ckpt = "google/vit-base-patch16-224-in21k"
        model_ckpt = "facebook/deit-base-distilled-patch16-224"
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(model_ckpt, do_resize=True)
        self.vit = transformers.DeiTModel.from_pretrained(model_ckpt) # return_dict=False

    def forward(self, x, a, r):
        x = self.image_processor(x, return_tensors="pt")
        x = self.vit(x.pixel_values)
        x = x.to_tuple()[0].squeeze()[0]
        x = torch.cat([x, a, r])
        x = x.unsqueeze(0)

        return x
