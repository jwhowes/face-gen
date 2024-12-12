from .util import BaseConfig, SubConfig
from typing import Optional, Dict


class VAEModelConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.dims = (64, 128, 256)
        self.depths = (2, 2, 1)
        self.d_latent = 4

        super().__init__(config)


class DiscriminatorConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.dims = (32, 64)
        self.depths = (2, 2)
        self.patch_size = 32

        super().__init__(config)


class VAEConfig(BaseConfig):
    def __init__(self, config_path: str):
        self.kl_weight = 0.1
        self.adv_weight = 0.1

        self.model = VAEModelConfig()
        self.discriminator = DiscriminatorConfig()

        super().__init__(config_path)
