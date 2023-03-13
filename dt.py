import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=1024, act_dim=4, image_dim=[3, 96, 96], device="cpu"):
        super().__init__()
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim

        config = transformers.DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim)
        self.transformer = transformers.DecisionTransformerModel(config).to(device=device)

        model_ckpt = "microsoft/beit-base-patch16-224-pt22k-ft22k"
        self.image_dim=image_dim
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(model_ckpt, do_resize=True, device=device)
        self.vit = transformers.BeitModel.from_pretrained(model_ckpt).to(device=device) # return_dict=False

    def forward(self, **kwargs):
        return self.transformer(**kwargs)

    def proc_state(self, o):
        o = torch.from_numpy(o).to(dtype=torch.float32, device=self.device).reshape(self.image_dim)
        o = self.image_processor(o, return_tensors="pt")
        e = self.vit(o.pixel_values.to(device=self.device))
        e = e.to_tuple()[0].squeeze()[0].to(device=self.device)

        return e
