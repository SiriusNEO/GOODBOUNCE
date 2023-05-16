import hydra
from big.registry import register

from isaacgymenvs.train import launch_rlg_hydra


@hydra.main(config_name="config", config_path="./big/env/cfg")
def launch_rlg_hydra_override(cfg):
    launch_rlg_hydra(cfg)


if __name__ == "__main__":
    register()
    launch_rlg_hydra_override()
