import yaml
import os
import torch

from typing import Optional, Dict, Tuple
from datetime import datetime

from . import accelerator


class SubConfig:
    def __init__(self, config: Optional[Dict] = None):
        if config is not None:
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)


class ModelConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.dims: Tuple[int] = (96, 192, 384, 768)
        self.depths: Tuple[int] = (2, 2, 5, 3)
        self.d_t: int = 384

        self.sigma_min: float = 1e-4

        super().__init__(config)


class DatasetConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.image_size: int = 192
        self.batch_size: int = 16

        super().__init__(config)


class Config(SubConfig):
    def unknown_tag(self, loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            constructor = loader.__class__.construct_scalar
        elif isinstance(node, yaml.SequenceNode):
            constructor = loader.__class__.construct_sequence
        elif isinstance(node, yaml.MappingNode):
            constructor = loader.__class__.construct_mapping
        else:
            raise NotImplementedError

        data = constructor(loader, node)

        return data

    def __init__(self, config_path: str, save: bool = False, metrics: Tuple[str] = ("loss",)):
        yaml.add_multi_constructor('!', self.unknown_tag)
        yaml.add_multi_constructor('tag:', self.unknown_tag)

        self.lr: float = 5e-5
        self.log_interval: int = 100
        self.num_epochs: int = 32
        self.warmup: float = 0.2
        self.clip_grad: Optional[float] = None

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super().__init__(config)
        self.lr = float(self.lr)

        if config is not None:
            self.model = ModelConfig(config["model"] if "model" in config else None)
            self.dataset = DatasetConfig(config["dataset"] if "dataset" in config else None)
        else:
            self.model = ModelConfig()
            self.dataset = DatasetConfig()

        self.exp_name = os.path.splitext(os.path.basename(config_path))[0]
        self.exp_dir = os.path.join("experiments", self.exp_name)

        if save and accelerator.is_main_process:
            if not os.path.isdir(self.exp_dir):
                os.makedirs(self.exp_dir)

            with open(os.path.join(self.exp_dir, "config.yaml"), "w+") as f:
                yaml.dump(self, f)

            with open(os.path.join(self.exp_dir, "log.csv"), "w+") as f:
                f.write(",".join(
                    ["epoch"] + list(metrics) + ["timestamp"]
                ) + "\n")

        self.epoch = 0

    def log(self, model, *metrics):
        if accelerator.is_main_process:
            torch.save(
                accelerator.get_state_dict(model),
                os.path.join(self.exp_dir, f"checkpoint_{self.epoch + 1:02}.pt")
            )

            with open(os.path.join(self.exp_dir, "log.csv"), "a") as f:
                f.write(",".join(
                    [str(self.epoch)] + [f"{m:.4f}" for m in metrics] + [str(datetime.now())]
                ) + "\n")
