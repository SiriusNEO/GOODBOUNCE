import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def parse(config_path: str):

    abs_path = to_absolute_path(config_path)
    global ret
    ret = None

    @hydra.main(config_name="config", config_path=abs_path)
    def __parse(cfg: DictConfig):
        # ensure checkpoints can be specified as relative paths
        if cfg.checkpoint:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        global ret
        ret = cfg

    __parse()

    return ret
    