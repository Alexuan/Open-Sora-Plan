from .modeling_causalvae import CausalVAEModel

from einops import rearrange
from torch import nn

class CausalVAEModelWrapper(nn.Module):
    # NOTE (Xuan)
    # def __init__(self, model_path, subfolder=None, cache_dir=None):
    def __init__(self, model_config, ckpt, subfolder=None, cache_dir=None):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        # NOTE (Xuan)
        # self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir)
        self.vae = CausalVAEModel.from_config(model_config)
        self.vae.init_from_ckpt(ckpt)
    def encode(self, x):  # b c t h w
        # x = self.vae.encode(x).sample()
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    def decode(self, x):
        # x = self.vae.decode(x)
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype
    #
    # def device(self):
    #     return self.vae.device