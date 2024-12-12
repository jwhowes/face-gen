import yaml
import os

from typing import Optional, Dict


class SubConfig:
    def __init__(self, config: Optional[Dict] = None):
        if config is not None:
            for k, v in config.items():
                if hasattr(self, k):
                    if isinstance(v, Dict):
                        getattr(self, k).__init__(v)
                    else:
                        setattr(self, k, v)


class DatasetConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.image_size = (160, 128)
        self.batch_size = 32

        super().__init__(config)


class BaseConfig(SubConfig):
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

    def __init__(self, config_path: str):
        yaml.add_multi_constructor('!', self.unknown_tag)
        yaml.add_multi_constructor('tag:', self.unknown_tag)

        self.lr = 5e-4
        self.num_epochs = 5
        self.clip_grad = 3.0
        self.log_interval = 100
        self.save_interval = 500

        self.dataset = DatasetConfig()

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.exp_name = os.path.splitext(os.path.basename(config_path))[0]

        super().__init__(config)

    def save(self):
        if not os.path.isdir(f"experiments/{self.exp_name}"):
            os.makedirs(f"experiments/{self.exp_name}")

        with open(f"experiments/{self.exp_name}/config.yaml", "w+") as f:
            yaml.dump(self, f)
