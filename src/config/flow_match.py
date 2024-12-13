from typing import Optional, Dict

from .util import BaseConfig, SubConfig


class FlowMatchModelConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.dims = (128, 256, 512)
        self.depths = (2, 2, 3)
        self.d_t = 512

        self.sigma_min = 1e-8

        super().__init__(config)


class LatentModelConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.dir = "experiments/vae"
        self.checkpoint = 2

        self.latent_scale = 1.7232484817504883

        super().__init__(config)


class FlowMatchConfig(BaseConfig):
    def __init__(self, config_path: str):
        self.latent_model = LatentModelConfig()
        self.model = FlowMatchModelConfig()

        super().__init__(config_path)
